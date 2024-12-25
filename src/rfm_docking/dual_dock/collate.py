import torch
from torch_geometric.data import Batch, HeteroData, Data

from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.reassignment import reassign_molecule
from rfm_docking.manifold_getter import DockingManifoldGetter
from rfm_docking.sampling import sample_harmonic_prior
from rfm_docking.featurization import get_feature_dims
from rfm_docking.utils import duplicate_and_rotate_tensors


class DualDockBatch(Batch):
    """Simply redefines the len method so we don't get into trouble when logging"""

    def __len__(self):
        return len(self.smiles)


def dual_dock_collate_fn(
    batch: list[HeteroData],
    manifold_getter: DockingManifoldGetter,
    do_ot: bool = False,
    sampling: str = "uniform",
) -> Batch:
    """Where the magic happens"""
    batch_size = len(batch)

    conformer = [i.conformer for i in batch]

    batch = Batch.from_data_list(batch)

    smiles = batch.smiles
    crystal_id = batch.crystal_id

    ####################################
    # First, we prepare the com docking
    osda = Batch.from_data_list(batch.osda)
    zeolite = Batch.from_data_list(batch.zeolite)

    com_batch = torch.repeat_interleave(torch.arange(batch_size), batch.loading)

    (
        com_x1,
        com_manifold,
        com_f_manifold,
        com_dims,
        com_mask_f,
    ) = manifold_getter(
        com_batch,
        osda.center_of_mass,
        split_manifold=True,
    )

    osda.dims = com_dims
    osda.mask_f = com_mask_f

    # sample osda in fractional space in georep (N, 3)
    com_x0 = com_manifold.random(
        *com_x1.shape, dtype=com_x1.dtype, device=com_x1.device
    )

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([osda.lengths, osda.angles], dim=-1)

    # potentially do ot
    if do_ot:
        # OT
        # in georep for reassignment:
        x0_geo = manifold_getter.flatrep_to_georep(com_x0, com_dims, com_mask_f).f
        x1_geo = manifold_getter.flatrep_to_georep(com_x1, com_dims, com_mask_f).f

        for i in range(batch_size):
            loading = int(batch.loading[i].item())

            x0_i = x0_geo[com_batch == i].view(loading, -1, 3)
            x1_i = x1_geo[com_batch == i].view(loading, -1, 3)

            # reassign x0 to x1
            _, row_ind, col_ind = reassign_molecule(x0_i, x1_i)
            # we reassign x0 because otherwise we would have to permute osda_x1 as well
            permuted_x0 = x0_i[row_ind]
            permuted_x0 = permuted_x0.view(-1, 3)
            x0_geo[com_batch == i] = permuted_x0

        com_x0 = manifold_getter.georep_to_flatrep(
            com_batch, x0_geo, split_manifold=True
        ).flat

    com_dock = Batch(
        x0=com_x0,
        x1=com_x1,
        osda=osda,
        zeolite=zeolite,
        atom_types=osda.atom_types,
        num_atoms=osda.num_atoms,
        manifold=com_manifold,
        f_manifold=com_f_manifold,
        mask_f=com_mask_f,
        batch=com_batch,
        dims=com_dims,
        lattices=lattices,
        y=batch.y,
        loading=batch.loading,
    )

    ##########################################
    # Second, we prepare the osda docking task
    conformer = duplicate_and_rotate_tensors(conformer, batch.loading)

    (
        osda_x1,
        osda_manifold,
        osda_f_manifold,
        osda_dims,
        osda_mask_f,
    ) = manifold_getter(
        osda.batch,
        osda.frac_coords,
        split_manifold=True,
    )

    osda.dims = osda_dims
    osda.mask_f = osda_mask_f

    osda_dock = Batch(
        x0=conformer,
        x1=osda_x1,
        osda=osda,
        zeolite=zeolite,
        atom_types=osda.atom_types,
        num_atoms=osda.num_atoms,
        manifold=osda_manifold,
        f_manifold=osda_f_manifold,
        mask_f=osda_mask_f,
        batch=osda.batch,
        dims=osda_dims,
        lattices=lattices,
        y=batch.y,
        loading=batch.loading,
    )

    batch = DualDockBatch(
        crystal_id=crystal_id,
        smiles=smiles,
        com_dock=com_dock,
        osda_dock=osda_dock,
    )

    return batch


class DualDockCollater:
    def __init__(
        self,
        manifold_getter: DockingManifoldGetter,
        do_ot: bool = False,
        sampling: str = "uniform",
    ):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot
        self.sampling = sampling

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return dual_dock_collate_fn(
            batch=batch,
            manifold_getter=self.manifold_getter,
            do_ot=self.do_ot,
            sampling=self.sampling,
        )
