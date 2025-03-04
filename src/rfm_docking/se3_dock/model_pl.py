"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from copy import deepcopy
import warnings
from functools import partial
from typing import Any, Literal

import time
import hydra
from hydra.utils import get_class
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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
from rfm_docking.manifm.solvers import projx_integrator, projx_integrator_return_last
from flowmm.model.solvers import (
    projx_cond_integrator_return_last,
    projx_integrate_xt_to_x1,
)
from flowmm.model.standardize import get_affine_stats
from rfm_docking.manifold_getter import SE3ManifoldGetter
from flowmm.rfm.vmap import VMapManifolds

from rfm_docking.se3_dock.se3_manifold import SE3ProductManifold
from rfm_docking.se3_dock.so3_utils import calc_rot_vf, rotvec_to_rotmat
from rfm_docking.se3_dock.integration import integrate_f_and_rot


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


class SE3DockingRFMLitModule(ManifoldFMLitModule):
    def __init__(self, cfg: DictConfig):
        pl.LightningModule.__init__(self)
        self.cfg = cfg
        self.save_hyperparameters()

        self.manifold_getter = SE3ManifoldGetter(
            dataset=cfg.data.dataset_name,
            **cfg.model.manifold_getter.manifolds,
        )
        self.manifold = SE3ProductManifold()

        self.costs = {
            "loss_f": cfg.model.cost_coord,
            "loss_rot": cfg.model.cost_rot,
            "loss_be": cfg.model.cost_be,
        }
        if cfg.model.affine_combine_costs:
            total_cost = sum([v for v in self.costs.values()])
            self.costs = {k: v / total_cost for k, v in self.costs.items()}

        model = hydra.utils.instantiate(
            self.cfg.vectorfield.vectorfield, _convert_="partial"
        )

        conjugate_model_class = get_class(self.cfg.vectorfield.conjugate_model._target_)
        # Model of the vector field.
        cspnet = conjugate_model_class(
            cspnet=model,
            manifold_getter=self.manifold_getter,
            coord_affine_stats=get_affine_stats(
                cfg.data.dataset_name, self.manifold_getter.coord_manifold
            ),
        )
        if cfg.optim.get("ema_decay", None) is None:
            self.model = cspnet
        else:
            self.model = EMA(
                cspnet,
                cfg.optim.ema_decay,
            )

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_metrics = {
            "loss": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_f": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_f": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_rot": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_rot": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "be_loss": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/be_loss": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
        }
        self.val_metrics = {
            "loss": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_f": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_f": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_rot": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_rot": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "be_loss": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/be_loss": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
        }
        self.test_metrics = {
            "loss": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_f": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_f": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_rot": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_rot": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "nll": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "be_loss": MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/be_loss": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
        }
        # for logging best so far validation accuracy
        self.val_metrics_best = {
            "loss": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_f": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_f": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "loss_rot": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/loss_rot": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "be_loss": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
            "unscaled/be_loss": MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            ),
        }
        if self.cfg.val.compute_nll:
            self.val_metrics["nll"] = MeanMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            )
            self.val_metrics_best["nll"] = MinMetric(
                compute_on_cpu=False, sync_on_compute=False, dist_sync_on_step=True
            )

    @staticmethod
    def _annealing_schedule(
        t: torch.Tensor, slope: float, intercept: float
    ) -> torch.Tensor:
        return slope * torch.nn.functional.relu(t - intercept) + 1.0

    @torch.no_grad()
    def sample(
        self,
        batch: HeteroData,
        num_steps: int = 1_000,
        entire_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.finish_sampling(
            batch=batch,
            f_0=batch.f_0,
            rot_0=batch.rot_0,
            manifold=self.manifold,
            num_steps=num_steps,
            entire_traj=entire_traj,
        )

    @torch.no_grad()
    def finish_sampling(
        self,
        batch: HeteroData,
        f_0: torch.Tensor,
        rot_0: torch.Tensor,
        manifold: VMapManifolds,
        num_steps: int,
        entire_traj: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        guidance_strength = self.cfg.integrate.get("guidance_strength", 0.0)
        print("Guidance strength is", guidance_strength)

        vecfield = partial(
            self.vecfield,
            batch=batch,  # NOTE assumes batch carries non-zero conditions
            guidance_strength=guidance_strength,
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
            f_t: torch.Tensor,
            rot_t: torch.Tensor,
            cond_coords: torch.Tensor | None = None,
            cond_be: torch.Tensor | None = None,
        ) -> torch.Tensor:
            anneal_factor = self._annealing_schedule(t, c, b)
            u_f_pred, rot_vec_pred, be_pred = vecfield(
                t=torch.atleast_2d(t),
                f_t=torch.atleast_2d(f_t),
                rot_t=torch.atleast_2d(rot_t),
                manifold=manifold,
                cond_coords=torch.atleast_2d(cond_coords)
                if isinstance(cond_coords, torch.Tensor)
                else cond_coords,
                cond_be=torch.atleast_2d(cond_be)
                if isinstance(cond_be, torch.Tensor)
                else cond_be,
            )
            if anneal_coords:
                # NOTE anneal only the coordinates, not the binding energy
                u_f_pred.mul_(anneal_factor)

            rotmat_pred = rotvec_to_rotmat(rot_vec_pred)
            u_rot_pred = calc_rot_vf(rot_t, rotmat_pred)

            return u_f_pred, u_rot_pred, be_pred

        if self.cfg.model.get("self_cond", False):  # TODO mrx ignoring for now
            """print("finish_sampling, self_cond True")
            x1 = projx_cond_integrator_return_last(
                manifold,
                scheduled_fn_to_integrate,
                f_0,
                rot_0,
                t=torch.linspace(0, 1, num_steps + 1).to(f_0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
                guidance_strength=guidance_strength,
            )
            return x1"""
            pass

        elif entire_traj or compute_traj_velo_norms:
            print(
                f"finish_sampling, {entire_traj} or {compute_traj_velo_norms}"
            )  # TODO mrx True, False
            f_s, v_f_s, rot_s, v_rot_s = integrate_f_and_rot(
                manifold,
                scheduled_fn_to_integrate,  # NOTE odefunc
                f_0,
                rot_0,
                t=torch.linspace(0, 1, num_steps + 1).to(f_0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                pbar=True,
            )
        else:
            """print("finish_sampling, else")  # TODO mrx ignore for now
            x1 = projx_integrator_return_last(
                manifold,
                scheduled_fn_to_integrate,
                f_0,
                rot_0,
                t=torch.linspace(0, 1, num_steps + 1).to(f_0.device),
                method=self.cfg.integrate.get("method", "euler"),
                projx=True,
                local_coords=False,
                pbar=True,
            )
            return x1"""
            pass

        if entire_traj:
            return f_s, v_f_s, rot_s, v_rot_s
        else:
            return f_s[-1], v_f_s[-1], rot_s[-1], v_rot_s[-1]

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
        raise NotImplementedError("compute_exact_loglikelihood not implemented yet.")

    def loss_fn(self, batch: Data, *args, **kwargs) -> dict[str, torch.Tensor]:
        return self.rfm_loss_fn(batch, *args, **kwargs)

    def rfm_loss_fn(
        self, batch: Data, x0: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        vecfield = partial(
            self.vecfield,
            batch=batch,
        )

        # [B, 3]
        f_0 = batch.f_0
        f_1 = batch.f_1

        # [B, 3, 3]
        rot_0 = batch.rot_0
        rot_1 = batch.rot_1

        # sample one t per datapoint, then extend to each molecule
        # [B', 1]
        t = torch.rand(
            batch.loading.shape[0], dtype=f_0.dtype, device=f_0.device
        ).reshape(-1, 1)

        t_per_mol = t[batch.batch]

        f_t, rot_t, u_f, u_rot = self.manifold(f_0, f_1, rot_0, rot_1, t_per_mol)

        u_f_pred, rot_vec_pred, be_pred = vecfield(
            t=t,
            f_t=f_t,
            rot_t=rot_t,
            manifold=self.manifold,
            cond_coords=None,
            cond_be=None,
        )

        rotmat_pred = rotvec_to_rotmat(rot_vec_pred)
        u_rot_pred = calc_rot_vf(rot_t, rotmat_pred)

        # loss for fractional coordinates
        # adjusted from FrameFlow https://arxiv.org/pdf/2401.04082 eqs. 20 and 21
        u_f_error = u_f_pred - u_f
        u_f_loss = (
            torch.sum(u_f_error**2, dim=-1)
            / (1 - torch.where(t_per_mol >= 0.9, 0.9, t_per_mol)) ** 2
        )
        u_f_loss = u_f_loss.mean()

        # loss for rotation
        u_rot_error = u_rot_pred - u_rot
        u_rot_loss = (
            torch.sum(u_rot_error**2, dim=-1)
            / (1 - torch.where(t_per_mol >= 0.9, 0.9, t_per_mol)) ** 2
        )
        u_rot_loss = u_rot_loss.mean()

        # loss for binding energy
        be_loss = F.l1_loss(be_pred, batch.y["bindingatoms"])

        loss = (
            self.costs["loss_f"] * u_f_loss
            + self.costs["loss_rot"] * u_rot_loss
            + self.costs["loss_be"] * be_loss
        )

        return {
            "loss": loss,
            "loss_f": self.costs["loss_f"] * u_f_loss,
            "unscaled/loss_f": u_f_loss,
            "loss_rot": self.costs["loss_rot"] * u_rot_loss,
            "unscaled/loss_rot": u_rot_loss,
            "be_loss": self.costs["loss_be"] * be_loss,
            "unscaled/be_loss": be_loss,
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
            torch.tensor(execution_time, device=v.device),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch),
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
            torch.tensor(execution_time, device=v.device),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(batch),
        )
        return out

    def compute_trajectory(
        self,
        batch: Data,
        num_steps: int = 1_000,
    ) -> dict[str, torch.Tensor | Data]:
        f_s, _, rot_s, _ = self.sample(batch, num_steps=num_steps, entire_traj=True)

        osda_atom_types = batch.osda.atom_types
        osda_frac_coords = batch.osda.frac_coords

        zeolite_atom_types = batch.zeolite.atom_types
        zeolite_frac_coords = batch.zeolite.frac_coords

        lattices = batch.lattices

        for i in torch.unique(batch.batch):
            mask = batch.batch == i

            osda = {
                "conformer_cart": batch.conformer[batch.osda.batch == i],
                "atom_types": osda_atom_types[batch.osda.batch == i],
                "frac_coords": osda_frac_coords[batch.osda.batch == i],
            }

            zeolite = {
                "atom_types": zeolite_atom_types[batch.zeolite.batch == i],
                "frac_coords": zeolite_frac_coords[batch.zeolite.batch == i],
            }

            out = {
                "crystal_id": batch.crystal_id[i],
                "loading": batch.loading[i],
                "f_s": f_s[:, mask].squeeze(),
                "rot_s": rot_s[:, mask],
                "lattices": lattices[i],
                "osda": osda,
                "zeolite": zeolite,
            }

            torch.save(out, f"{batch.crystal_id[i]}_traj.pt")

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
                batch_size=self.cfg.data.datamodule.batch_size.val,
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
        if self.cfg.integrate.get("entire_traj", False):
            print("predict_step compute_trajectory")
            out = self.compute_trajectory(
                batch,
                num_steps=self.cfg.integrate.get("num_steps", 1_000),
            )
        else:
            """print("predict_step compute_generation")  # TODO mrx ignore for now
            out = self.compute_generation(
                batch,
                dim_coords=self.cfg.data.get("dim_coords", 3),
                num_steps=self.cfg.integrate.get("num_steps", 1_000),
            )"""
            pass

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
