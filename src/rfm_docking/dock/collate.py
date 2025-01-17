import torch
from torch_geometric.data import Batch, HeteroData, Data
from torch_geometric.loader.dataloader import Collater

from rfm_docking.reassignment import reassign_molecule
from rfm_docking.manifold_getter import DockingManifoldGetter
from rfm_docking.sampling import (
    sample_uniform_then_gaussian,
    sample_uniform,
    sample_uniform_then_conformer,
    sample_voronoi,
    get_sigma,
)
from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.utils import duplicate_and_rotate_tensors


def dock_collate_fn(
    batch: list[HeteroData],
    manifold_getter: DockingManifoldGetter,
    do_ot: bool = False,
    sampling="normal",
) -> HeteroData:
    """Where the magic happens"""

    conformers = [i.conformer for i in batch]

    batch = Batch.from_data_list(batch)

    # batch data
    y = batch.y  # NOTE dictionary of properties
    osda = batch.osda
    zeolite = batch.zeolite

    smiles = batch.smiles
    crystal_id = batch.crystal_id

    osda = Batch.from_data_list(osda)
    zeolite = Batch.from_data_list(zeolite)

    (
        x1,
        osda_manifold,
        osda_f_manifold,
        osda_dims,
        osda_mask_f,
    ) = manifold_getter(
        osda.batch,
        osda.frac_coords,
        split_manifold=True,
    )

    osda.manifold = osda_manifold
    osda.f_manifold = osda_f_manifold
    osda.dims = osda_dims
    osda.mask_f = osda_mask_f

    _, _, _, zeolite_dims, zeolite_mask_f = manifold_getter(
        zeolite.batch, zeolite.frac_coords, split_manifold=True
    )
    zeolite.dims = zeolite_dims
    zeolite.mask_f = zeolite_mask_f

    def sample(sampling):
        if "sampling" == "normal":
            x0 = osda_manifold.random(*x1.shape, dtype=x1.dtype, device=x1.device)
            return x0
        elif sampling == "uniform":
            x0 = torch.rand_like(osda.frac_coords)
        elif sampling == "uniform_then_gaussian":
            sigma = get_sigma(
                sigma_in_A=3, lattice_lenghts=osda.lengths, num_atoms=osda.num_atoms
            )
            x0 = sample_uniform_then_gaussian(osda, batch.loading, sigma=sigma)
        elif sampling == "uniform_then_conformer":
            conformer = duplicate_and_rotate_tensors(conformers, batch.loading)
            conformer = manifold_getter.georep_to_flatrep(
                osda.batch, conformer, split_manifold=True
            ).flat
            conformer = osda_manifold.projx(conformer)
            conformer = manifold_getter.flatrep_to_georep(
                conformer, osda_dims, osda_mask_f
            ).f
            osda.conformer = conformer
            x0 = sample_uniform_then_conformer(osda, smiles, batch.loading)
        elif "voronoi" in sampling:
            sigma = get_sigma(
                sigma_in_A=3, lattice_lenghts=osda.lengths, num_atoms=osda.num_atoms
            )
            x0 = sample_voronoi(
                osda.num_atoms,
                osda.batch,
                zeolite.voronoi_nodes,
                zeolite.num_voronoi_nodes,
                loading=batch.loading,
            )
            if sampling == "voronoi_then_gaussian":
                # sample from a Gaussian distribution around the Voronoi nodes
                x0 += torch.randn_like(x0) * sigma
            elif sampling == "voronoi_then_conformer":
                conformer = duplicate_and_rotate_tensors(conformers, batch.loading)
                conformer = manifold_getter.georep_to_flatrep(
                    osda.batch, conformer, split_manifold=True
                ).flat
                conformer = osda_manifold.projx(conformer)
                conformer = manifold_getter.flatrep_to_georep(
                    conformer, osda_dims, osda_mask_f
                ).f
                x0 += conformer
            else:
                raise ValueError(f"Voronoi sampling <{sampling}> not recognized")
        else:
            raise ValueError(f"Sampling method <{sampling}> not recognized")

        # (N, 3) -> (N*3, )
        x0 = manifold_getter.georep_to_flatrep(osda.batch, x0, False).flat

        x0 = osda_manifold.projx(x0)
        return x0

    x0 = sample(sampling=sampling)

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([osda.lengths, osda.angles], dim=-1)

    # potentially do ot
    if do_ot:
        # OT
        # in georep for reassignment:
        x0_geo = manifold_getter.flatrep_to_georep(x0, osda_dims, osda_mask_f).f
        x1_geo = manifold_getter.flatrep_to_georep(x1, osda_dims, osda_mask_f).f

        # iterate over batch
        for i in range(len(batch)):
            loading = int(batch.loading[i].item())

            x0_i = x0_geo[osda.batch == i].view(loading, -1, 3)
            x1_i = x1_geo[osda.batch == i].view(loading, -1, 3)

            # reassign x0 to x1
            permuted_x0, _, _ = reassign_molecule(x0_i, x1_i)
            permuted_x0 = permuted_x0.view(-1, 3)
            x0_geo[osda.batch == i] = permuted_x0

        x0 = manifold_getter.georep_to_flatrep(
            osda.batch, x0_geo, split_manifold=True
        ).flat

    batch = Batch(
        crystal_id=crystal_id,
        smiles=smiles,
        osda=osda,
        zeolite=zeolite,
        loading=batch.loading,
        x0=x0,
        x1=x1,
        lattices=lattices,
        num_atoms=osda.num_atoms,
        manifold=osda_manifold,
        f_manifold=osda_f_manifold,
        dims=osda_dims,
        mask_f=osda_mask_f,
        batch=osda.batch,
        y=y,
    )
    return batch


class DockCollater:
    def __init__(
        self,
        manifold_getter: DockingManifoldGetter,
        do_ot: bool = False,
        sampling: str = "normal",
    ):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot
        self.sampling = sampling

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return dock_collate_fn(batch, self.manifold_getter, self.do_ot, self.sampling)
