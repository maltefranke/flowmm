"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import torch
from geoopt import Manifold
from torch import nn
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import HeteroData

from torch_scatter import scatter

from diffcsp.common.data_utils import radius_graph_pbc
from diffcsp.common.data_utils import (
    lattice_params_to_matrix_torch,
)
from diffcsp.pl_modules.cspnet import CSPLayer as DiffCSPLayer
from diffcsp.pl_modules.cspnet import CSPNet as DiffCSPNet
from diffcsp.pl_modules.cspnet import SinusoidsEmbedding
from diffcsp.script_utils import chemical_symbols

from flowmm.data import NUM_ATOMIC_TYPES
from flowmm.rfm.manifold_getter import (
    Dims,
    ManifoldGetter,
    ManifoldGetterOut,
)
from flowmm.rfm.manifolds.flat_torus import FlatTorus01


class CSPLayer(DiffCSPLayer):
    """Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim,
        act_fn,
        dis_emb,
        ln,
        n_space: int = 3,
        represent_num_atoms: bool = False,
        self_cond: bool = False,
    ):
        nn.Module.__init__(self)

        self.self_cond = self_cond
        self.n_space = n_space
        self.dis_emb = dis_emb

        if dis_emb is None:
            self.dis_dim = n_space
        else:
            self.dis_dim = dis_emb.dim
        if self_cond:
            self.dis_dim *= 2

        self.represent_num_atoms = represent_num_atoms
        if represent_num_atoms:
            self.one_hot_dim = 100  # largest cell of atoms that we'd represent, this is safe for a HACK
            self.num_atom_embedding = nn.Linear(
                self.one_hot_dim, hidden_dim, bias=False
            )
            num_hidden_dim_vecs = 3
        else:
            num_hidden_dim_vecs = 2

        self.edge_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim * num_hidden_dim_vecs
                + self.dis_dim
                + 3 * 2,  # NOTE: 3*2 to use invariant representation of lattice
                hidden_dim,
            ),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        lattices,
        edge_index,
        edge2graph,
        frac_diff,
        num_atoms,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        edge_features = []

        if self.dis_emb is not None:
            if self.self_cond:
                _frac_diff, _pred_frac_diff = torch.tensor_split(frac_diff, 2, dim=-1)
                _frac_diff = self.dis_emb(_frac_diff)
                _pred_frac_diff = (
                    torch.zeros_like(_frac_diff)
                    if (torch.zeros_like(_pred_frac_diff) == _pred_frac_diff).all()
                    else self.dis_emb(_pred_frac_diff)
                )
                frac_diff = torch.concat([_frac_diff, _pred_frac_diff], dim=-1)
            else:
                frac_diff = self.dis_emb(frac_diff)

        lattices_flat = lattices.reshape(lattices.shape[0], -1)
        lattices_flat_edges = lattices_flat[edge2graph]

        edge_features.extend([hi, hj, lattices_flat_edges, frac_diff])

        return self.edge_mlp(torch.cat(edge_features, dim=1))

    def forward(
        self,
        node_features,
        lattices,
        edge_index,
        edge2graph,
        frac_diff,
        num_atoms: torch.LongTensor,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features,
            lattices,
            edge_index,
            edge2graph,
            frac_diff,
            num_atoms,
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class DockAndOptimizeCSPNet(DiffCSPNet):
    def __init__(
        self,
        hidden_dim: int = 512,
        time_dim: int = 256,
        num_layers: int = 6,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        n_space: int = 3,
        num_freqs: int = 128,
        edge_style: str = "fc",
        cutoff: float = 7.0,
        max_neighbors: int = 20,
        ln: bool = True,
        use_log_map: bool = True,
        dim_atomic_rep: int = NUM_ATOMIC_TYPES,
        self_edges: bool = True,
        self_cond: bool = False,
    ):
        nn.Module.__init__(self)

        self.n_space = n_space
        self.time_emb = nn.Linear(1, time_dim, bias=False)

        self.self_cond = self_cond
        if self_cond:
            coef = 2
        else:
            coef = 1

        self.node_embedding = nn.Embedding(len(chemical_symbols), hidden_dim)

        self.atom_latent_emb = nn.Linear(
            hidden_dim + time_dim,
            hidden_dim,
            bias=True,  # False
        )

        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=n_space)
        elif dis_emb == "none":
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "dock_layer_%d" % i,
                CSPLayer(
                    hidden_dim,
                    self.act_fn,
                    self.dis_emb,
                    ln=ln,
                    n_space=n_space,
                    self_cond=self_cond,
                ),
            )

        for i in range(0, num_layers):
            self.add_module(
                "optimize_layer_%d" % i,
                CSPLayer(
                    hidden_dim,
                    self.act_fn,
                    self.dis_emb,
                    ln=ln,
                    n_space=n_space,
                    self_cond=self_cond,
                ),
            )

        self.num_layers = num_layers

        # it makes sense to have no bias here since p(F) is translation invariant
        self.coord_out = nn.Linear(hidden_dim, n_space, bias=False)

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.ln = ln
        self.edge_style = edge_style
        self.use_log_map = use_log_map
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_edges = self_edges

    def gen_edges(
        self, num_atoms, frac_coords, lattices, node2graph, edge_style="knn", radius=7.0
    ):
        if edge_style == "fc":
            if self.self_edges:
                lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
            else:
                lis = [
                    torch.ones(n, n, device=num_atoms.device)
                    - torch.eye(n, device=num_atoms.device)
                    for n in num_atoms
                ]
            fc_graph = torch.block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            fc_edges = fc_edges.to(frac_coords.device)

            if self.use_log_map:
                # this is the shortest torus distance, but DiffCSP didn't use it
                frac_diff = FlatTorus01.logmap(
                    frac_coords[fc_edges[0]], frac_coords[fc_edges[1]]
                )
            else:
                frac_diff = frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]
            return fc_edges, frac_diff

        elif edge_style == "knn":
            _lattices = lattice_params_to_matrix_torch(lattices[:, :3], lattices[:, 3:])
            lattice_nodes = _lattices[node2graph]
            cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords,
                None,
                None,
                num_atoms.to(cart_coords.device),
                radius,
                self.max_neighbors,
                device=cart_coords.device,
                lattices=_lattices,
            )

            if self.use_log_map:
                # this is the shortest torus distance, but DiffCSP didn't use it
                # not sure it makes sense for the cartesian space version
                distance_vectors = FlatTorus01.logmap(
                    frac_coords[edge_index[0]], frac_coords[edge_index[1]]
                )
            else:
                distance_vectors = (
                    frac_coords[edge_index[1]] - frac_coords[edge_index[0]]
                )
            distance_vectors += to_jimages.float()

            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
                edge_index, to_jimages, num_bonds, distance_vectors
            )

            return edge_index_new, -edge_vector_new

    def forward(
        self,
        batch: HeteroData,
        t,
    ):
        t_emb = self.time_emb(t)
        t_emb = t_emb.expand(
            batch.num_atoms.shape[0], -1
        )  # if there is a single t, repeat for the batch

        # create graph
        edges, frac_diff = self.gen_edges(
            batch.num_atoms,
            batch.frac_coords,
            batch.lattices,
            batch.batch,
            edge_style="knn",  # NOTE fc = fully connected
            radius=self.cutoff,
        )
        edge2graph = batch.batch[edges[0]]

        # neural network
        # embed atom features
        node_features = self.node_embedding(batch.atom_types)
        t_per_atom = t_emb.repeat_interleave(batch.num_atoms.to(t_emb.device), dim=0)

        node_features = torch.cat(
            [
                node_features,
                t_per_atom,
            ],
            dim=1,
        )
        node_features = self.atom_latent_emb(node_features)
        node_features = self.act_fn(node_features)

        # do docking first
        for i in range(0, self.num_layers):
            # update osda node feats
            node_features = self._modules["dock_layer_%d" % i](
                node_features,
                batch.lattices,
                edges,
                edge2graph,
                frac_diff,
                batch.num_atoms,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # predict coords
        coord_out = self.coord_out(node_features)

        return coord_out


class ProjectedConjugatedCSPNet(nn.Module):
    def __init__(
        self,
        cspnet: DockAndOptimizeCSPNet,
        manifold_getter: ManifoldGetter,
        coord_affine_stats: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.cspnet = cspnet
        self.manifold_getter = manifold_getter
        self.self_cond = cspnet.self_cond
        self.metric_normalized = False

        if coord_affine_stats is not None:
            self.register_buffer(
                "coord_u_t_mean", coord_affine_stats["u_t_mean"].unsqueeze(0)
            )
            self.register_buffer(
                "coord_u_t_std", coord_affine_stats["u_t_std"].unsqueeze(0)
            )

    def _conjugated_forward(
        self,
        batch: HeteroData,
        t: torch.Tensor,
        x: torch.Tensor,
        cond: torch.Tensor | None,
    ) -> ManifoldGetterOut:
        frac_coords = self.manifold_getter.flatrep_to_georep(
            x,
            dims=batch.dims,
            mask_f=batch.mask_f,
        )
        batch.frac_coords = frac_coords.f

        if self.self_cond:
            if cond is not None:
                fc_cond = self.manifold_getter.flatrep_to_georep(
                    cond,
                    dims=batch.dims,
                    mask_f=batch.mask_f,
                )
                fc_cond = fc_cond.f

            else:
                fc_cond = torch.zeros_like(frac_coords)

            batch.frac_coords = torch.cat([batch.frac_coords, fc_cond], dim=-1)

        coord_out = self.cspnet(
            batch,
            t,
        )

        return self.manifold_getter.georep_to_flatrep(
            batch=batch.batch,
            frac_coords=coord_out,
            split_manifold=False,
        )

    def forward(
        self,
        batch: HeteroData,
        t: torch.Tensor,
        x: torch.Tensor,
        manifold: Manifold,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """u_t: [0, 1] x M -> T M

        representations are mapped as follows:
        `flat -> flat_manifold -> pytorch_geom -(nn)-> pytorch_geom -> flat_tangent_estimate -> flat_tangent`
        """
        x = manifold.projx(x)
        if cond is not None:
            cond = manifold.projx(cond)
        v, *_ = self._conjugated_forward(
            batch,
            t,
            x,
            cond,
        )
        # NOTE comment out to predict position directly
        v = manifold.proju(x, v)

        if self.metric_normalized and hasattr(manifold, "metric_normalized"):
            v = manifold.metric_normalized(x, v)
        return v
