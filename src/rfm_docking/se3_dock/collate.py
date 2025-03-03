import torch
from torch_geometric.data import Batch, HeteroData, Data

from rfm_docking.reassignment import reassign_molecule
from rfm_docking.manifold_getter import SE3ManifoldGetter
from rfm_docking.sampling import (
    sample_voronoi,
)
from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.utils import duplicate_and_rotate_tensors, sample_rotation_matrices
from rfm_docking.product_space_dock.utils import (
    rigid_transform_Kabsch_3D_torch_batch,
    matrix_to_axis_angle,
)
from diffcsp.common.data_utils import (
    lattice_params_to_matrix_torch,
    frac_to_cart_coords,
    cart_to_frac_coords,
)


def se3_dock_collate_fn(
    batch: list[HeteroData],
    manifold_getter: SE3ManifoldGetter,
    do_ot: bool = False,
    sampling="normal",
) -> HeteroData:
    """
    Collate function for the SE3 docking dataset. We do the following:

    1. randomly rotate the conformers
    2. calculate ground truth rotation rot_1 with the Kabsch algorithm
    3. sample random rotation matrices rot_0 and center of masses f_0
    """

    conformers_cart = [i.conformer for i in batch]
    wrapped_coords_cart = [i.osda.wrapped_coords_cart for i in batch]

    batch = Batch.from_data_list(batch)

    # match the conformers to get the ground truth rotation matrices
    rot_1 = []
    all_conf_cart = []
    for i, (conf, wrapped) in enumerate(zip(conformers_cart, wrapped_coords_cart)):
        conf_batched = conf.repeat(batch.loading[i], 1, 1)

        # TODO for debugging, always take the same conformer
        # rotate the conformers randomly
        random_rotations = sample_rotation_matrices(
            batch.loading[i], dtype=conf_batched.dtype
        )
        conf_cart_rotated = torch.einsum("bij,bnj->bni", random_rotations, conf_batched)

        all_conf_cart.append(conf_cart_rotated.reshape(-1, 3))

        wrapped_cart_batched = wrapped.reshape(batch.loading[i], -1, 3)

        # calculate the ground truth rotation matrix with the kabsch algorithm
        rot_1_i, _ = rigid_transform_Kabsch_3D_torch_batch(
            conf_cart_rotated, wrapped_cart_batched
        )

        rot_1.append(rot_1_i)

    rot_1 = torch.cat(rot_1)
    conf_cart_rotated = torch.cat(all_conf_cart, dim=0)

    # batch data
    y = batch.y  # NOTE dictionary of properties
    osda = batch.osda
    zeolite = batch.zeolite

    smiles = batch.smiles
    crystal_id = batch.crystal_id

    osda = Batch.from_data_list(osda)
    zeolite = Batch.from_data_list(zeolite)

    f_1 = osda.center_of_mass

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
        elif sampling == "com":
            f_0 = osda.center_of_mass
        else:
            raise ValueError(f"Sampling method <{sampling}> not recognized")

        # sample random rotation matrices from scipy
        num_matrices = batch.loading.sum()
        rot_0 = sample_rotation_matrices(num_matrices, dtype=conf_batched.dtype)

        return f_0, rot_0

    f_0, rot_0 = sample(sampling=sampling)

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([osda.lengths, osda.angles], dim=-1)

    # potentially do ot
    """if do_ot:
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
        ).flat"""

    batch_id = torch.repeat_interleave(
        torch.arange(batch.loading.shape[0], device=f_0.device), batch.loading
    )

    batch = Batch(
        crystal_id=crystal_id,
        smiles=smiles,
        osda=osda,
        zeolite=zeolite,
        loading=batch.loading,
        f_0=f_0,
        f_1=f_1,
        rot_0=rot_0,
        rot_1=rot_1,
        conformer=conf_cart_rotated,
        lattices=lattices,
        num_atoms=osda.num_atoms,
        batch=batch_id,
        y=y,
    )
    return batch


class SE3DockCollater:
    def __init__(
        self,
        manifold_getter: SE3ManifoldGetter,
        do_ot: bool = False,
        sampling: str = "normal",
    ):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot
        self.sampling = sampling

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return se3_dock_collate_fn(
            batch, self.manifold_getter, self.do_ot, self.sampling
        )
