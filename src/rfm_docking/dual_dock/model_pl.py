"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Literal

import time
import hydra
from hydra.utils import get_class
import pytorch_lightning as pl
import torch
from scipy.spatial.transform import Rotation as R
from geoopt.manifolds.euclidean import Euclidean
from geoopt.manifolds.product import ProductManifold
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.func import jvp, vjp
from torch_geometric.data import Data, Batch, HeteroData
from torchmetrics import MeanMetric, MinMetric

from diffcsp.common.data_utils import (
    lattices_to_params_shape,
    lattice_params_to_matrix_torch,
)
from manifm.ema import EMA
from manifm.model_pl import ManifoldFMLitModule, div_fn, output_and_div
from manifm.solvers import projx_integrator, projx_integrator_return_last
from flowmm.model.solvers import (
    projx_cond_integrator_return_last,
    projx_integrate_xt_to_x1,
)
from flowmm.model.standardize import get_affine_stats
from rfm_docking.manifold_getter import DockingManifoldGetter, Dims
from flowmm.rfm.manifolds.spd import SPDGivenN, SPDNonIsotropicRandom
from flowmm.rfm.vmap import VMapManifolds
from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.reassignment import ot_reassignment
from rfm_docking.sampling import get_sigma


def output_and_div(
    vecfield: callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    v: torch.Tensor | None = None,
    div_mode: Literal["exact", "rademacher"] = "exact",
) -> tuple[torch.Tensor, torch.Tensor]:
    if div_mode == "exact":
        dx = vecfield(x)
        div = div_fn(vecfield)(x)
    else:
        dx, vjpfunc = vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class DualDockingRFMLitModule(ManifoldFMLitModule):
    def __init__(self, cfg: DictConfig):
        pl.LightningModule.__init__(self)
        self.cfg = cfg
        self.save_hyperparameters()

        self.manifold_getter = DockingManifoldGetter(
            coord_manifold=cfg.model.manifold_getter.coord_manifold,
            dataset=cfg.data.dataset_name,
        )

        self.costs = {
            "loss_f": cfg.model.cost_coord,
        }
        if cfg.model.affine_combine_costs:
            total_cost = sum([v for v in self.costs.values()])
            self.costs = {k: v / total_cost for k, v in self.costs.items()}

        com_dock_model = hydra.utils.instantiate(
            self.cfg.vectorfield.com_dock_vectorfield, _convert_="partial"
        )

        conjugate_model_class = get_class(self.cfg.vectorfield.com_dock_model._target_)
        # Model of the vector field.
        com_dock_cspnet = conjugate_model_class(
            cspnet=com_dock_model,
            manifold_getter=self.manifold_getter,
            coord_affine_stats=get_affine_stats(
                cfg.data.dataset_name, self.manifold_getter.coord_manifold
            ),
        )

        osda_dock_cspnet = hydra.utils.instantiate(
            self.cfg.vectorfield.osda_dock_vectorfield, _convert_="partial"
        )

        conjugate_model_class = get_class(self.cfg.vectorfield.osda_dock_model._target_)
        # Model of the vector field.
        osda_dock_cspnet = conjugate_model_class(
            cspnet=osda_dock_cspnet,
            manifold_getter=self.manifold_getter,
            coord_affine_stats=get_affine_stats(
                cfg.data.dataset_name, self.manifold_getter.coord_manifold
            ),
        )

        self.model = None
        if cfg.optim.get("ema_decay", None) is None:
            self.com_dock_model = com_dock_cspnet
            self.osda_dock_model = osda_dock_cspnet
        else:
            self.com_dock_model = EMA(
                com_dock_cspnet,
                cfg.optim.ema_decay,
            )
            self.osda_dock_model = EMA(
                osda_dock_cspnet,
                cfg.optim.ema_decay,
            )

        # https://github.com/Lightning-AI/pytorch-lightning/issues/18803
        metrics_kwargs = {
            "compute_on_cpu": False,
            "sync_on_compute": False,
            "dist_sync_on_step": True,
        }
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = {
            "loss": MeanMetric(**metrics_kwargs),
            "loss_f": MeanMetric(**metrics_kwargs),
            "unscaled/loss_f": MeanMetric(**metrics_kwargs),
            "loss_be": MeanMetric(**metrics_kwargs),
        }
        self.val_metrics = {
            "loss": MeanMetric(**metrics_kwargs),
            "loss_f": MeanMetric(**metrics_kwargs),
            "unscaled/loss_f": MeanMetric(**metrics_kwargs),
            "loss_be": MeanMetric(**metrics_kwargs),
        }
        self.test_metrics = {
            "loss": MeanMetric(**metrics_kwargs),
            "loss_f": MeanMetric(**metrics_kwargs),
            "unscaled/loss_f": MeanMetric(**metrics_kwargs),
            "loss_be": MeanMetric(**metrics_kwargs),
            "nll": MeanMetric(**metrics_kwargs),
        }
        # for logging best so far validation accuracy
        self.val_metrics_best = {
            "loss": MinMetric(**metrics_kwargs),
            "loss_f": MinMetric(**metrics_kwargs),
            "loss_be": MinMetric(**metrics_kwargs),
            "unscaled/loss_f": MinMetric(**metrics_kwargs),
        }
        if self.cfg.val.compute_nll:
            self.val_metrics["nll"] = MeanMetric()
            self.val_metrics_best["nll"] = MinMetric()

    @property
    def device(self):
        return self.com_dock_model.parameters().__next__().device

    @staticmethod
    def _annealing_schedule(
        t: torch.Tensor, slope: float, intercept: float
    ) -> torch.Tensor:
        return slope * torch.nn.functional.relu(t - intercept) + 1.0

    @torch.no_grad()
    def sample(
        self,
        batch: HeteroData,
        x0: torch.Tensor = None,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            x1,
            manifold,
            f_manifold,
            dims,
            mask_f,
        ) = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            split_manifold=True,
        )
        if x0 is None:
            x0 = manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)
        else:
            x0 = x0.to(x1)

        return self.finish_sampling(
            batch=batch,
            x0=x0,
            manifold=manifold,
            f_manifold=f_manifold,
            dims=dims,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def gen_sample(
        self,
        batch: HeteroData,
        dim_coords: int,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shape,
            manifold,
            f_manifold,
            dims,
            mask_f,
        ) = self.manifold_getter.from_empty_batch(
            batch.com_dock.batch, dim_coords, split_manifold=True
        )
        num_atoms = self.manifold_getter._get_num_atoms(mask_f)

        x0 = batch.com_dock.x0

        return self.finish_sampling(
            batch=batch,
            x0=x0,
            manifold=manifold,
            f_manifold=f_manifold,
            dims=dims,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def pred_sample(
        self,
        batch: HeteroData,
        atom_types: torch.LongTensor,
        dim_coords: int,
        x0: torch.Tensor = None,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            shape,
            manifold,
            f_manifold,
            dims,
            mask_f,
        ) = self.manifold_getter.from_only_atom_types(
            batch.batch, atom_types, dim_coords, split_manifold=True
        )
        num_atoms = self.manifold_getter._get_num_atoms(mask_f)

        if x0 is None:
            x0 = manifold.random(*shape, device=batch.batch.device)
        else:
            x0 = x0.to(device=batch.batch.device)

        return self.finish_sampling(
            batch=batch,
            x0=x0,
            manifold=manifold,
            f_manifold=f_manifold,
            dims=dims,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def finish_sampling(
        self,
        batch: HeteroData,
        x0: torch.Tensor,
        manifold: VMapManifolds,
        f_manifold: VMapManifolds,
        dims: Dims,
        num_steps: int,
        entire_traj: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        com_dock = batch.com_dock
        osda_dock = batch.osda_dock

        # translate the conformer to com_dock_x1_pred
        atoms_per_mol = torch.repeat_interleave(
            batch.osda_dock.num_atoms // batch.osda_dock.loading,
            batch.osda_dock.loading,
        )

        com_dock_vecfield = partial(
            self.com_dock_model,
            batch=com_dock,
        )

        osda_dock_vecfield = partial(
            self.osda_dock_model,
            batch=osda_dock,
        )

        compute_traj_velo_norms = self.cfg.integrate.get(
            "compute_traj_velo_norms", False
        )

        c = self.cfg.integrate.get("inference_anneal_slope", 0.0)
        b = self.cfg.integrate.get("inference_anneal_offset", 0.0)

        anneal_coords = self.cfg.integrate.get("inference_anneal_coords", True)

        print(
            "Anneal Coords:",
            anneal_coords,
        )

        def scheduled_fn_to_integrate(
            t: torch.Tensor,
            x: torch.Tensor,
            vecfield,
            manifold,
            cond_coords: torch.Tensor | None = None,
            cond_be: torch.Tensor | None = None,
        ) -> torch.Tensor:
            anneal_factor = self._annealing_schedule(t, c, b)
            coord_out, _ = vecfield(
                t=torch.atleast_2d(t),
                x=torch.atleast_2d(x),
                manifold=manifold,
                cond_coords=torch.atleast_2d(cond_coords)
                if isinstance(cond_coords, torch.Tensor)
                else cond_coords,
                cond_be=torch.atleast_2d(cond_be)
                if isinstance(cond_be, torch.Tensor)
                else cond_be,
            )
            if anneal_coords:
                coord_out[:, 0 : dims.f].mul_(anneal_factor)
            return coord_out

        if self.cfg.model.get("self_cond", False):
            com_dock_x1_pred = projx_cond_integrator_return_last(
                com_dock.manifold,
                partial(
                    scheduled_fn_to_integrate,
                    vecfield=com_dock_vecfield,
                    manifold=com_dock.manifold,
                ),
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )

            # set the docked structure as x0 for the optimization task
            com_dock_x1_pred_expanded = torch.repeat_interleave(
                com_dock_x1_pred, atoms_per_mol, dim=0
            )

            osda_dock_x0 = osda_dock.x0 + com_dock_x1_pred_expanded
            osda_dock_x0 = self.manifold_getter.georep_to_flatrep(
                batch.osda_dock.batch, osda_dock_x0, split_manifold=True
            ).flat

            x1 = projx_cond_integrator_return_last(
                osda_dock.manifold,
                partial(
                    scheduled_fn_to_integrate,
                    vecfield=osda_dock_vecfield,
                    manifold=osda_dock.manifold,
                ),
                osda_dock_x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )

            return x1

        elif entire_traj or compute_traj_velo_norms:
            com_dock_traj, vs = projx_integrator(
                com_dock.manifold,
                partial(
                    scheduled_fn_to_integrate,
                    vecfield=com_dock_vecfield,
                    manifold=com_dock.manifold,
                ),
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                pbar=True,
            )

            # set the docked structure as x0 for the optimization task
            com_dock_x1_pred_expanded = torch.repeat_interleave(
                self.manifold_getter.flatrep_to_georep(
                    com_dock_traj[-1], com_dock.dims, com_dock.mask_f
                ).f,
                atoms_per_mol,
                dim=0,
            )

            osda_dock_x0 = osda_dock.x0 + com_dock_x1_pred_expanded
            osda_dock_x0 = self.manifold_getter.georep_to_flatrep(
                batch.osda_dock.batch, osda_dock_x0, split_manifold=True
            ).flat
            osda_dock_x0 = osda_dock.manifold.projx(osda_dock_x0)

            osda_dock_traj, vs = projx_integrator(
                osda_dock.manifold,
                partial(
                    scheduled_fn_to_integrate,
                    vecfield=osda_dock_vecfield,
                    manifold=osda_dock.manifold,
                ),
                osda_dock_x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                pbar=True,
            )

        else:
            com_dock_x1_pred_expanded = projx_integrator_return_last(
                com_dock.manifold,
                partial(
                    scheduled_fn_to_integrate,
                    vecfield=com_dock_vecfield,
                    manifold=com_dock.manifold,
                ),
                x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )

            # set the docked structure as x0 for the optimization task
            com_dock_x1_pred_expanded = torch.repeat_interleave(
                com_dock_x1_pred, atoms_per_mol, dim=0
            )

            osda_dock_x0 = osda_dock.x0 + com_dock_x1_pred_expanded
            osda_dock_x0 = self.manifold_getter.georep_to_flatrep(
                batch.osda_dock.batch, osda_dock_x0, split_manifold=True
            ).flat

            x1 = projx_integrator_return_last(
                osda_dock.manifold,
                partial(
                    scheduled_fn_to_integrate,
                    vecfield=osda_dock_vecfield,
                    manifold=osda_dock.manifold,
                ),
                osda_dock_x0,
                t=torch.linspace(0, 1, num_steps + 1).to(x0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )
            return x1

        if compute_traj_velo_norms:
            s = 0
            e = dims.f
            norm_f = f_manifold.inner(
                xs[..., s:e], vs[..., s:e], vs[..., s:e], data_in_dim=1
            )

        if entire_traj and compute_traj_velo_norms:
            # return xs, norm_a, norm_f, norm_l
            return xs, norm_f
        elif entire_traj and not compute_traj_velo_norms:
            return com_dock_traj, osda_dock_traj
        elif not entire_traj and compute_traj_velo_norms:
            # return xs[0], norm_a, norm_f, norm_l
            return xs[0], norm_f
        else:
            # this should happen due to logic above
            return xs[0]

    @torch.no_grad()
    def compute_exact_loglikelihood(
        self,
        batch: Data,
        stage: str,
        t1: float = 1.0,
        return_projx_error: bool = False,
        num_steps: int = 1_000,
    ):
        """Computes the negative log-likelihood of a batch of data."""
        x1, manifold, dims, mask_f = self.manifold_getter(
            batch.osda.batch,
            batch.osda.atom_types,
            batch.osda.frac_coords,
            batch.osda.lengths,
            batch.osda.angles,
            split_manifold=False,
        )
        dim = sum(dims)

        nfe = [0]

        div_mode = self.cfg.integrate.get(
            "div_mode", "rademacher"
        )  # alternative: exact

        v = None
        if div_mode == "rademacher":
            v = torch.randint(low=0, high=2, size=x1.shape).to(x1) * 2 - 1

        vecfield = partial(self.vecfield, batch=batch)

        def odefunc(t, tensor):
            nfe[0] += 1
            t = t.to(tensor)
            x = tensor[..., :dim]

            def l_vecfield(x):
                return vecfield(
                    t=torch.atleast_2d(t), x=torch.atleast_2d(x), manifold=manifold
                )

            dx, div = output_and_div(l_vecfield, x, v=v, div_mode=div_mode)

            if hasattr(manifold, "logdetG"):
                corr = jvp(manifold.logdetG, (x,), (dx,))[1]
                div = div + 0.5 * corr.to(div)

            div = div.reshape(-1, 1)
            del t, x
            return torch.cat([dx, div], dim=-1)

        # Solve ODE on the product manifold of data manifold x euclidean.
        prod_manis = [
            ProductManifold((m, dim), (Euclidean(), 1)) for m in manifold.manifolds
        ]
        product_man = VMapManifolds(prod_manis)
        state1 = torch.cat([x1, torch.zeros_like(x1[..., :1])], dim=-1)

        with torch.no_grad():
            state0 = projx_integrator_return_last(
                product_man,
                odefunc,
                state1,
                t=torch.linspace(t1, 0, num_steps + 1).to(x1),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )

        x0, logdetjac = state0[..., :dim], state0[..., -1]

        x0_ = x0
        x0 = manifold.projx(x0)

        # log how close the final solution is to the manifold.
        integ_error = (x0[..., :dim] - x0_[..., :dim]).abs().max()
        # self.log(f"{stage}/integ_error", integ_error, batch_size=batch.batch_size)

        logp0 = manifold.base_logprob(x0)
        logp1 = logp0 + logdetjac

        if self.cfg.get("normalize_loglik", True):
            logp1 = logp1 / batch.num_atoms

        # Mask out those that left the manifold
        masked_logp1 = logp1
        if isinstance(manifold, (SPDGivenN, SPDNonIsotropicRandom)):
            mask = integ_error < 1e-5
            masked_logp1 = logp1[mask]

        if return_projx_error:
            return logp1, integ_error
        else:
            return masked_logp1

    def loss_fn(self, batch: Data, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.rfm_loss_fn(batch, *args, **kwargs)

    def rfm(self, batch: Data, model, calculate_x1: bool = False):
        vecfield = partial(
            model,
            batch=batch,
        )

        x0 = batch.x0
        x1 = batch.x1

        manifold = batch.manifold

        N = x1.shape[0]

        t = torch.rand(N, dtype=x1.dtype, device=x1.device).reshape(-1, 1)

        x0 = manifold.projx(x0)
        x1 = manifold.projx(x1)

        x_t, u_t = manifold.cond_u(x0, x1, t)
        x_t = x_t.reshape(N, x0.shape[-1])
        u_t = u_t.reshape(N, x0.shape[-1])

        # this projects out the mean from the tangent vectors
        # our model cannot predict it, so keeping it in inflates the loss
        u_t = manifold.proju(x_t, u_t)

        cond_coords = None
        cond_be = None
        if self.cfg.model.self_cond:
            with torch.no_grad():
                if torch.rand((1)) < 0.5:
                    cond_coords, cond_be = projx_integrate_xt_to_x1(
                        manifold,
                        lambda t, x: vecfield(
                            t=torch.atleast_2d(t),
                            x=torch.atleast_2d(x),
                            manifold=manifold,
                        ),
                        x_t,
                        t,
                        method=self.cfg.integrate.get("method", "euler"),
                    ).detach_()

        u_t_pred, be_pred = vecfield(
            t=t,
            x=x_t,
            manifold=manifold,
            cond_coords=cond_coords,
            cond_be=cond_be,
        )
        # maybe adjust the target
        diff = u_t_pred - u_t

        if calculate_x1:
            with torch.no_grad():
                x1_pred = projx_integrate_xt_to_x1(
                    manifold,
                    lambda t, x: vecfield(
                        t=torch.atleast_2d(t),
                        x=torch.atleast_2d(x),
                        manifold=manifold,
                    ),
                    x_t,
                    t,
                    vt=u_t_pred,
                    method="euler",
                )
        else:
            x1_pred = None

        max_num_atoms = batch.mask_f.size(-1)
        dim_f_per_atom = batch.dims.f / max_num_atoms

        s = 0
        e = batch.dims.f
        # loss for each example in batch
        loss_f = (
            batch.f_manifold.inner(x_t[:, s:e], diff[:, s:e], diff[:, s:e])
            / dim_f_per_atom
        )
        loss_f = loss_f.mean()
        # per dim, already per atom

        be_loss = torch.nn.functional.mse_loss(be_pred, batch.y["bindingatoms"])

        return loss_f, u_t_pred, be_loss, x1_pred

    def rfm_loss_fn(
        self, batch: Data, dock_x0: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        #####################
        # First, we prepare the center of mass docking task
        com_docking_loss_f, _, com_be_loss, com_dock_x1_pred = self.rfm(
            batch.com_dock, model=self.com_dock_model, calculate_x1=True
        )

        """com_dock_x1_pred = self.manifold_getter.flatrep_to_georep(
            com_dock_x1_pred, batch.com_dock.dims, batch.com_dock.mask_f
        ).f"""
        #####################
        # Second, we prepare the osda docking task
        # randomly rotated conformers w/ mean 0 are saved in batch.osda_dock.x0

        # translate the conformer to com_dock_x1_pred
        atoms_per_mol = torch.repeat_interleave(
            batch.osda_dock.num_atoms // batch.osda_dock.loading,
            batch.osda_dock.loading,
        )
        """com_dock_x1_pred_expanded = torch.repeat_interleave(
            com_dock_x1_pred, atoms_per_mol, dim=0
        )"""

        com_expanded = torch.repeat_interleave(
            batch.osda_dock.osda.center_of_mass, atoms_per_mol, dim=0
        )
        # noise the center of mass
        sigma = get_sigma(
            sigma_in_A=3,
            lattice_lenghts=batch.osda_dock.osda.lengths,
            num_atoms=batch.osda_dock.num_atoms,
        )
        com_expanded += torch.randn_like(com_expanded) * sigma

        batch.osda_dock.x0 += com_expanded
        batch.osda_dock.x0 = self.manifold_getter.georep_to_flatrep(
            batch.osda_dock.batch, batch.osda_dock.x0, split_manifold=True
        ).flat

        osda_docking_loss_f, _, osda_be_loss, _ = self.rfm(
            batch.osda_dock, model=self.osda_dock_model
        )

        loss_f = 0.5 * com_docking_loss_f + 0.5 * osda_docking_loss_f
        # loss_f = com_docking_loss_f
        # loss_be = 0.5 * osda_be_loss + 0.5 * com_be_loss

        loss = self.costs["loss_f"] * loss_f  # + loss_be

        return {
            "loss": loss,
            "loss_f": self.costs["loss_f"] * loss_f,
            "unscaled/loss_f": loss_f,
            # "loss_be": loss_be,
        }

    def training_step(self, batch: Data, batch_idx: int):
        start_time = time.time()
        loss_dict = self.loss_fn(batch)

        if torch.isfinite(loss_dict["loss"]):
            # log train metrics

            for k, v in loss_dict.items():
                self.log(
                    f"train/{k}",
                    v,
                    batch_size=self.cfg.data.datamodule.batch_size.train,
                )
                self.train_metrics[k].update(v.cpu())
        else:
            # skip step if loss is NaN.
            print(f"Skipping iteration because loss is {loss_dict['loss'].item()}.")
            return None

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        end_time = time.time()
        execution_time = (end_time - start_time) / len(batch)

        # Log the execution time per example
        self.log(
            "train_step_time_per_example",
            execution_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return loss_dict

    def training_epoch_end(self, outputs: list[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for train_metric in self.train_metrics.values():
            train_metric.reset()

    def shared_eval_step(
        self,
        batch: Data,
        batch_idx: int,
        stage: Literal["val", "test"],
        compute_loss: bool,
        compute_nll: bool,
    ) -> dict[str, torch.Tensor]:
        start_time = time.time()
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        out = {}

        if compute_loss:
            loss_dict = self.loss_fn(batch)
            if stage == "val":
                batch_size = self.cfg.data.datamodule.batch_size.val
            else:
                batch_size = self.cfg.data.datamodule.batch_size.test

            for k, v in loss_dict.items():
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )
                metrics[k].update(v.cpu())
            out.update(loss_dict)

        if compute_nll:
            nll_dict = {}
            logprob = self.compute_exact_loglikelihood(
                batch,
                stage,
                num_steps=self.cfg.integrate.get("num_steps", 1_000),
            )
            nll = -logprob.mean()
            nll_dict[f"{stage}/nll"] = nll
            nll_dict[f"{stage}/nll_num_steps"] = self.cfg.integrate.num_steps

            self.logger.log_metrics(nll_dict, step=self.global_step)
            metrics["nll"].update(nll.cpu())
            out.update(nll_dict)

        end_time = time.time()
        execution_time = (end_time - start_time) / len(batch)

        # Log the execution time per example
        self.log(
            f"{stage}_step_time_per_example",
            execution_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return out

    def compute_reconstruction(
        self,
        batch: HeteroData,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            split_manifold=False,
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.sample(batch, num_steps=num_steps)
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.sample(batch, num_steps=num_steps)
            norms = {}
        atom_types, frac_coords, lattices = self.manifold_getter.flatrep_to_crystal(
            recon, dims, mask_f
        )
        lengths, angles = lattices_to_params_shape(lattices)
        out = {
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattices": lattices,
            "lengths": lengths,
            "angles": angles,
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_recon_trajectory(
        self,
        batch: Data,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter(
            batch.batch,
            batch.atom_types,
            batch.frac_coords,
            batch.lengths,
            batch.angles,
            split_manifold=False,
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.sample(
                batch, num_steps=num_steps, entire_traj=True
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.sample(batch, num_steps=num_steps, entire_traj=True)
            norms = {}

        atom_types, frac_coords, lattices = [], [], []
        lengths, angles = [], []
        for recon_step in recon:
            f = self.manifold_getter.flatrep_to_crystal(recon_step, dims, mask_f)
            # atom_types.append(batch.atom_types)
            frac_coords.append(f)
            lattices.append(batch.lattice)
            lengths.append(batch.lengths)
            angles.append(batch.angles)
        out = {
            "atom_types": torch.stack(atom_types, dim=0),
            "frac_coords": torch.stack(frac_coords, dim=0),
            "lattices": torch.stack(lattices, dim=0),
            "lengths": torch.stack(lengths, dim=0),
            "angles": torch.stack(angles, dim=0),
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch,
        }
        out.update(norms)
        return out

    def compute_generation(
        self,
        batch: Data,
        dim_coords: int = 3,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter.from_empty_batch(
            batch.batch, dim_coords, split_manifold=False
        )

        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.gen_sample(
                batch, dim_coords, num_steps=num_steps
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.gen_sample(batch, dim_coords, num_steps=num_steps)
            norms = {}
        frac_coords = self.manifold_getter.flatrep_to_crystal(recon, dims, mask_f)
        out = {
            "atom_types": batch.atom_types,
            "frac_coords": frac_coords,
            "lattices": batch.lattices,
            "lengths": batch.lattices[:, :3],
            "angles": batch.lattices[:, 3:],
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch.batch,
        }
        out.update(norms)
        return out

    def compute_gen_trajectory(
        self,
        batch: Data,
        dim_coords: int = 3,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.gen_sample(
                batch.batch,
                dim_coords,
                num_steps=num_steps,
                entire_traj=True,
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            com_dock_traj, osda_dock_traj = self.gen_sample(
                batch, dim_coords, num_steps=num_steps, entire_traj=True
            )
            norms = {}

        com_traj, osda_traj, zeolite_traj = [], [], []
        for com_dock_recon_step in com_dock_traj:
            f = self.manifold_getter.flatrep_to_crystal(
                com_dock_recon_step, batch.com_dock.dims, batch.com_dock.mask_f
            )
            com_traj.append(f)
            zeolite_traj.append(batch.com_dock.zeolite.frac_coords)

        for osda_dock_recon_step in osda_dock_traj:
            f = self.manifold_getter.flatrep_to_crystal(
                osda_dock_recon_step, batch.osda_dock.dims, batch.osda_dock.mask_f
            )

            osda_traj.append(f)

        com_traj = torch.stack(com_traj, dim=0)
        osda_traj = torch.stack(osda_traj, dim=0)
        zeolite_traj = torch.stack(zeolite_traj, dim=0)

        for i in range(batch.com_dock.batch.max() + 1):
            com_dict = {
                "atom_types": torch.ones((batch.com_dock.batch == i).sum()),
                "target_coords": batch.com_dock.osda.center_of_mass[
                    batch.com_dock.batch == i
                ],
                "frac_coords": com_traj[:, batch.com_dock.batch == i],
            }

            osda_dict = {
                "atom_types": batch.osda_dock.osda.atom_types[
                    batch.osda_dock.batch == i
                ],
                "target_coords": batch.osda_dock.osda.frac_coords[
                    batch.osda_dock.batch == i
                ],
                "frac_coords": osda_traj[:, batch.osda_dock.batch == i],
            }

            zeolite_dict = {
                "atom_types": batch.com_dock.zeolite.atom_types[
                    batch.com_dock.zeolite.batch == i
                ],
                "target_coords": batch.com_dock.zeolite.frac_coords[
                    batch.com_dock.zeolite.batch == i
                ],
                "frac_coords": zeolite_traj[:, batch.com_dock.zeolite.batch == i],
            }

            out = {
                "crystal_id": batch.crystal_id[i],
                "smiles": batch.smiles[i],
                "loading": batch.com_dock.loading[i],
                "com": com_dict,
                "osda": osda_dict,
                "zeolite": zeolite_dict,
                "lattices": batch.com_dock.lattices[i],
            }
            out.update(norms)
            torch.save(out, f"{batch.crystal_id[i]}_traj.pt")

        return out

    def compute_prediction(
        self,
        batch: Data,
        dim_coords: int = 3,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        *_, dims, mask_f = self.manifold_getter.from_empty_batch(
            batch.batch, dim_coords, split_manifold=False
        )
        if self.cfg.integrate.get("compute_traj_velo_norms", False):
            recon, norms_a, norms_f, norms_l = self.pred_sample(
                batch.batch, batch.atom_types, dim_coords, num_steps=num_steps
            )
            norms = {"norms_a": norms_a, "norms_f": norms_f, "norms_l": norms_l}
        else:
            recon = self.pred_sample(
                batch.batch, batch.atom_types, dim_coords, num_steps=num_steps
            )
            norms = {}
        frac_coords = self.manifold_getter.flatrep_to_crystal(recon, dims, mask_f)
        out = {
            "atom_types": batch.atom_types,
            "frac_coords": frac_coords,
            "lattices": batch.lattices,
            "lengths": batch.lengths,
            "angles": batch.angles,
            "num_atoms": batch.num_atoms,
            "input_data_batch": batch.batch,
        }
        out.update(norms)
        return out

    def validation_step(self, batch: Data, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="val",
            compute_loss=True,
            compute_nll=self.cfg.val.compute_nll,
        )

    def validation_epoch_end(self, outputs: list[Any]):
        out = {}
        for key, val_metric in self.val_metrics.items():
            val_metric_value = (
                val_metric.compute()
            )  # get val accuracy from current epoch
            val_metric_best = self.val_metrics_best[key]
            val_metric_best.update(val_metric_value)
            self.log(
                f"val/best/{key}",
                val_metric_best.compute(),
                on_epoch=True,
                prog_bar=True,
            )
            val_metric.reset()
            out[key] = val_metric_value
        return out

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_eval_step(
            batch,
            batch_idx,
            stage="test",
            compute_loss=self.cfg.test.get("compute_loss", False),
            compute_nll=self.cfg.test.get("compute_nll", False),
        )

    def test_epoch_end(self, outputs: list[Any]):
        for test_metric in self.test_metrics.values():
            test_metric.reset()

    def predict_step(self, batch: Any, batch_idx: int):
        start_time = time.time()
        if not hasattr(batch, "frac_coords"):
            if self.cfg.integrate.get("entire_traj", False):
                out = self.compute_gen_trajectory(
                    batch,
                    dim_coords=self.cfg.data.get("dim_coords", 3),
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )
            else:
                out = self.compute_generation(
                    batch,
                    dim_coords=self.cfg.data.get("dim_coords", 3),
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )
        else:
            # not generating or predicting new structures
            if self.cfg.integrate.get("entire_traj", False):
                out = self.compute_recon_trajectory(
                    batch,
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )
            else:
                out = self.compute_reconstruction(
                    batch,
                    num_steps=self.cfg.integrate.get("num_steps", 1_000),
                )

        end_time = time.time()
        execution_time = (end_time - start_time) / len(batch)

        # NOTE logging not available in predict_step :/
        out["predict_time_per_example"] = execution_time
        return out

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        if self.cfg.optim.get("lr_scheduler", None) is not None:
            lr_scheduler = hydra.utils.instantiate(
                self.cfg.optim.lr_scheduler,
                optimizer,
            )
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": "epoch",
                        "monitor": self.cfg.optim.monitor,
                        "frequency": self.cfg.optim.frequency,
                    },
                }
            elif isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": self.cfg.optim.interval,
                    },
                }
            else:
                raise NotImplementedError("unsuported lr_scheduler")
        else:
            return {"optimizer": optimizer}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.com_dock_model, EMA):
            self.com_dock_model.update_ema()
        if isinstance(self.osda_dock_model, EMA):
            self.osda_dock_model.update_ema()
