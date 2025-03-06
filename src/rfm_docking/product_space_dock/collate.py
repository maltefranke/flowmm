import torch
from torch_geometric.data import Batch, HeteroData, Data
from torch_geometric.loader.dataloader import Collater

from rfm_docking.reassignment import reassign_molecule
from rfm_docking.manifold_getter import ProductSpaceManifoldGetter
from rfm_docking.sampling import (
    sample_voronoi,
)
from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.utils import duplicate_and_rotate_tensors
from rfm_docking.product_space_dock.utils import (
    sample_sphere,
    modify_conformer_torsion_angles,
    axis_angle_to_matrix,
)


def product_space_dock_collate_fn(
    batch: list[HeteroData],
    manifold_getter: ProductSpaceManifoldGetter,
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
        flat_1,
        manifold,
        f_manifold,
        rot_manifold,
        tor_manifold,
        dims,
        mask_f,
        mask_tor,
    ) = manifold_getter(
        frac_coords=osda.frac_coords,
        rot=osda.ro,
        tor=osda.tor,
        coord_batch=osda.batch,
        tor_batch=osda.tor_batch,
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
        # sample the center of masses
        if sampling == "uniform":
            f_0 = torch.rand_like(osda.center_of_mass)
        elif sampling == "voronoi":
            f_0 = sample_voronoi(
                osda.loading,
                osda.batch,
                zeolite.voronoi_nodes,
                zeolite.num_voronoi_nodes,
                loading=batch.loading,
            )
        else:
            raise ValueError(f"Sampling method <{sampling}> not recognized")

        # sample rotation vector
        rot_0 = sample_sphere(osda.loading.sum())

        # sample torsion angles
        tor_0 = torch.rand_like(osda.tor)

        return f_0, rot_0, tor_0

    f_0, rot_0, tor_0 = sample(sampling=sampling)

    conformer = duplicate_and_rotate_tensors(conformers, batch.loading)

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

    x0 = (f_0, rot_0, tor_0)
    x1 = (f_1, rot_1, tor_1)

    manifolds = (f_manifold, rot_manifold, tor_manifold)
    masks = (mask_f, mask_rot, mask_tor)
    dims = (f_dims, rot_dims, tor_dims)

    batch = Batch(
        crystal_id=crystal_id,
        smiles=smiles,
        osda=osda,
        zeolite=zeolite,
        loading=batch.loading,
        x0=x0,
        x1=x1,
        conformer=conformer,
        lattices=lattices,
        num_atoms=osda.num_atoms,
        manifolds=manifolds,
        dims=dims,
        masks=masks,
        batch=osda.batch,
        y=y,
    )
    return batch


class ProductSpaceDockCollater:
    def __init__(
        self,
        manifold_getter: ProductSpaceManifoldGetter,
        do_ot: bool = False,
        sampling: str = "normal",
    ):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot
        self.sampling = sampling

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return product_space_dock_collate_fn(
            batch, self.manifold_getter, self.do_ot, self.sampling
        )
