import torch
from torch_geometric.data import Batch, HeteroData, Data

from rfm_docking.manifold_getter import DockingManifoldGetter
from rfm_docking.sampling import (
    get_sigma,
)
from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01


def optimize_collate_fn(
    batch: list[HeteroData],
    manifold_getter: DockingManifoldGetter,
    do_ot: bool = False,
    sampling="uniform",
    stage="train",
) -> HeteroData:
    """Where the magic happens"""

    batch = Batch.from_data_list(batch)

    frac_coords = [
        torch.cat([osda_opt.frac_coords, zeolite_opt.frac_coords], dim=0)
        for osda_opt, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]

    atom_types = [
        torch.cat([osda_opt.atom_types, zeolite_opt.atom_types], dim=0)
        for osda_opt, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]

    edges = [osda_opt.edge_index for osda_opt in batch.osda_opt]
    edge_feats = [osda_opt.edge_feats for osda_opt in batch.osda_opt]

    node_feats = [
        torch.cat([osda_opt.node_feats, zeolite_opt.node_feats], dim=0)
        for osda_opt, zeolite_opt in zip(batch.osda_opt, batch.zeolite_opt)
    ]
    lengths = [osda.lengths for osda in batch.osda_opt]
    angles = [osda.angles for osda in batch.osda_opt]

    data_list = [
        Data(
            frac_coords=FlatTorus01.projx(frac_coords[i]),  # project to 0 to 1
            atom_types=atom_types[i],
            node_feats=node_feats[i],
            edge_index=edges[i],
            edge_feats=edge_feats[i],
            lengths=lengths[i],
            angles=angles[i],
            num_atoms=batch.num_atoms[i],
            batch=torch.ones_like(frac_coords[i], dtype=torch.long) * i,
        )
        for i in range(len(batch))
    ]

    optimize_batch = Batch.from_data_list(data_list)

    (
        x1,
        optimize_manifold,
        optimize_f_manifold,
        optimize_dims,
        optimize_mask_f,
    ) = manifold_getter(
        optimize_batch.batch,
        optimize_batch.frac_coords,
        split_manifold=True,
    )

    osda = batch.osda
    zeolite = batch.zeolite

    smiles = batch.smiles
    crystal_id = batch.crystal_id

    osda = Batch.from_data_list(osda)
    zeolite = Batch.from_data_list(zeolite)

    x0 = [
        torch.cat([osda_dock.frac_coords, zeolite_dock.frac_coords], dim=0)
        for osda_dock, zeolite_dock in zip(batch.osda, batch.zeolite)
    ]
    x0 = torch.cat(x0, dim=0)

    # only during training add some noise to make the model more robust
    if stage == "train":
        sigma = get_sigma(
            sigma_in_A=0.1,
            lattice_lenghts=osda.lengths,
            num_atoms=optimize_batch.num_atoms,
        )
        x0 += torch.randn_like(x0) * sigma

    x0 = manifold_getter.georep_to_flatrep(optimize_batch.batch, x0, False).flat

    x0 = FlatTorus01.projx(x0)

    # lattices is the invariant(!!) representation of the lattice, parametrized by lengths and angles
    lattices = torch.cat([osda.lengths, osda.angles], dim=-1)

    batch = Batch(
        crystal_id=crystal_id,
        smiles=smiles,
        osda=osda,
        zeolite=zeolite,
        loading=batch.loading,
        x0=x0,
        x1=x1,
        lattices=lattices,
        node_feats=optimize_batch.node_feats,
        num_atoms=optimize_batch.num_atoms,
        manifold=optimize_manifold,
        f_manifold=optimize_f_manifold,
        dims=optimize_dims,
        mask_f=optimize_mask_f,
        batch=optimize_batch.batch,
        y=batch.y,
    )
    return batch


class OptimizeCollater:
    def __init__(
        self,
        manifold_getter: DockingManifoldGetter,
        do_ot: bool = False,
        sampling: str = "uniform",
        stage: str = "train",
    ):
        self.manifold_getter = manifold_getter
        self.do_ot = do_ot
        self.sampling = sampling
        self.stage = stage

    def __call__(self, batch: list[HeteroData]) -> HeteroData:
        return optimize_collate_fn(
            batch, self.manifold_getter, self.do_ot, self.sampling, self.stage
        )
