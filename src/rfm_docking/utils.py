import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.utils import dense_to_sparse
from sklearn.cluster import DBSCAN
from pymatgen.core.lattice import Lattice
from scipy.spatial.transform import Rotation as R

from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from diffcsp.common.data_utils import radius_graph_pbc
from diffcsp.common.data_utils import lattice_params_to_matrix_torch


def smiles_to_pos(smiles, forcefield="mmff", device="cpu"):
    """Convert smiles to 3D coordinates."""
    # Use RDKit to generate a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    # Generate 3D coordinates for the molecule
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)

    if forcefield == "mmff":
        AllChem.MMFFOptimizeMolecule(mol)
    elif forcefield == "uff":
        AllChem.UFFOptimizeMolecule(mol)
    else:
        raise ValueError("Unrecognised force field")

    mol = Chem.RemoveHs(mol)

    # Extract the atom symbols and coordinates
    atom_coords = []
    for i, _ in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        pos = torch.tensor(
            [positions.x, positions.y, positions.z], dtype=torch.float32, device=device
        )
        atom_coords.append(pos)

    atom_coords = torch.stack(atom_coords)

    # remove mean to center the molecule
    atom_coords -= atom_coords.mean(0)

    return atom_coords


def sample_rotation_matrices(num_matrices: torch.Tensor) -> torch.Tensor:
    """Samples random 3D rotation matrices."""
    return torch.tensor(
        R.random(num_matrices).as_matrix(), dtype=torch.float32
    )  # Shape: (num_matrices, 3, 3)


def duplicate_and_rotate_tensors(
    tensors: list[torch.Tensor], loadings: torch.Tensor
) -> torch.Tensor:
    """
    For each tensor, duplicate it based on its loading, sample rotation matrices,
    and apply the rotations to the duplicated tensor.
    """
    rotated_tensors = []

    for tensor, loading in zip(tensors, loadings):
        # Duplicate the tensor
        duplicated_tensor = tensor.repeat(loading, 1, 1)  # Shape: (loading, X_i, 3)

        # Sample rotation matrices
        rotation_matrices = sample_rotation_matrices(loading)  # Shape: (loading, 3, 3)

        # Apply rotation matrices
        rotated_copies = torch.einsum(
            "nij,nmj->nmi", rotation_matrices, duplicated_tensor
        ).reshape(-1, 3)

        rotated_tensors.append(rotated_copies)

    rotated_tensors = torch.cat(rotated_tensors, dim=0)
    return rotated_tensors


def gen_edges(
    num_atoms,
    frac_coords,
    lattices,
    node2graph,
    edge_style="knn",
    radius=7.0,
    max_neighbors=15,
    self_edges=False,
):
    if edge_style == "fc":
        if self_edges:
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

        # this is the shortest torus distance, but DiffCSP didn't use it
        frac_diff = FlatTorus01.logmap(
            frac_coords[fc_edges[0]], frac_coords[fc_edges[1]]
        )
        return fc_edges, frac_diff

    elif edge_style == "knn":
        _lattices = lattice_params_to_matrix_torch(lattices[:, :3], lattices[:, 3:])
        lattice_nodes = _lattices[node2graph]
        cart_coords = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)

        # we are dealing with huge graphs, so we need to loop over each graph to reduce memory usage
        all_edges = []
        all_num_bonds = []
        for i, lattice in enumerate(_lattices):
            edge_index_i, _, num_bonds = radius_graph_pbc(
                cart_coords[node2graph == i],
                None,
                None,
                num_atoms[i].view(-1).to(cart_coords.device),
                radius,
                max_neighbors,
                device=cart_coords.device,
                lattices=lattice.view(1, 3, 3),
            )
            all_edges.append(edge_index_i)
            all_num_bonds.append(num_bonds)

        all_edges = [edges + num_atoms[:i].sum() for i, edges in enumerate(all_edges)]
        edge_index = torch.cat(all_edges, dim=1)
        num_bonds = torch.cat(all_num_bonds, dim=0)

        # this is the shortest torus distance, but DiffCSP didn't use it
        # not sure it makes sense for the cartesian space version
        edge_vector = FlatTorus01.logmap(
            frac_coords[edge_index[0]], frac_coords[edge_index[1]]
        )

        return edge_index, edge_vector


def merge_edges(num_osda, num_zeolite, osda_edges, zeolite_edges):
    osda_edges = osda_edges.clone()
    zeolite_edges = zeolite_edges.clone()

    # osda offsets
    osda_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=num_osda.device), num_osda]), dim=0
    )[:-1]

    # zeolite offsets
    zeolite_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], device=num_zeolite.device), num_zeolite]), dim=0
    )[:-1]

    zeolite_indices = [
        torch.logical_and(
            zeolite_edges >= zeolite_offsets[i], zeolite_edges < zeolite_offsets[i + 1]
        )
        for i in range(len(num_osda) - 1)
    ]
    zeolite_indices.append(zeolite_edges >= zeolite_offsets[-1])

    osda_indices = [
        torch.logical_and(
            osda_edges >= osda_offsets[i], osda_edges < osda_offsets[i + 1]
        )
        for i in range(len(num_osda) - 1)
    ]
    osda_indices.append(osda_edges >= osda_offsets[-1])

    for idx, (zeo_i, osda_i) in enumerate(zip(zeolite_indices, osda_indices)):
        zeolite_edges[zeo_i] += osda_offsets[idx] + num_osda[idx]
        osda_edges[osda_i] += zeolite_offsets[idx]
    return torch.cat([osda_edges, zeolite_edges], dim=1)


def fast_wrap_coords_edge_based(
    osda_pos: torch.Tensor,
    lattice: torch.Tensor,
    edge_idx: torch.Tensor,
    return_cart: bool = True,
) -> list[torch.Tensor]:
    """
    Wrap coordinates to unit cell.
    1. Cluster the osda atoms spatially
    2. For each cluster, translate atoms with edges to the closest position
    """
    possible_translations = torch.tensor(
        [
            [-1, -1, -1],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, 0, -1],
            [-1, 0, 0],
            [-1, 0, 1],
            [-1, 1, -1],
            [-1, 1, 0],
            [-1, 1, 1],
            [0, -1, -1],
            [0, -1, 0],
            [0, -1, 1],
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, -1, -1],
            [1, -1, 0],
            [1, -1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.int32,
    )
    lattice = Lattice(lattice)
    # cluster the osda atoms spatially
    clustering = DBSCAN(eps=1.65, min_samples=1).fit(osda_pos)

    labels = torch.tensor(clustering.labels_, dtype=torch.int32)
    unique_labels = torch.unique(labels)

    indices = torch.arange(osda_pos.shape[0])
    wrapped_positions = []

    for cluster in unique_labels:
        # all the cluster atoms don't need to be updated anymore
        already_updated_nodes = set(indices[labels == cluster].tolist())
        queue = set(indices[labels == cluster].tolist())

        updated_coords = osda_pos.clone()
        frac_coords = lattice.get_fractional_coords(updated_coords)
        frac_coords = torch.tensor(frac_coords, dtype=torch.float32)

        while len(queue) > 0:
            new_queue = set()
            # iterate over the queue, which is a list of neighbors from the last node
            for i in queue:
                # determine the neighbors of i
                edges = edge_idx[1, edge_idx[0] == i]

                # remove already updated nodes
                edges = edges[edges not in already_updated_nodes]

                # if all neighbors have already been updated, go to the next node in queue
                if len(edges) == 0:
                    continue

                # move connections as close as possible to i
                # only +/- 1 and 0 allowed to make updates wrt. periodicity
                coords_to_translate = updated_coords[edges, :].view(-1, 3)
                coords_to_translate = lattice.get_fractional_coords(coords_to_translate)
                coords_to_translate = torch.tensor(
                    coords_to_translate, dtype=torch.float32
                )

                translated_coords = possible_translations.unsqueeze(
                    0
                ) + coords_to_translate.unsqueeze(1)

                # take the closest position to the original position
                distances = torch.norm(translated_coords - frac_coords[i], dim=-1)
                min_idx = torch.min(distances, -1)

                frac_coords[edges] += possible_translations[min_idx[1]]
                updated_coords = lattice.get_cartesian_coords(frac_coords)
                updated_coords = torch.tensor(updated_coords, dtype=torch.float32)

                # mark node as updated. i.e. this node will not be updated again
                already_updated_nodes.update({(i)})

                unvisited_connected_nodes = [
                    x
                    for x in edges.flatten().tolist()
                    if x not in already_updated_nodes
                ]
                new_queue.update(unvisited_connected_nodes)

            # update the queue
            queue = new_queue - already_updated_nodes

        wrapped_positions.append(updated_coords)

        # stop because we're only interested in the mean
        break

    wrapped_positions = wrapped_positions[0]
    if return_cart:
        return wrapped_positions
    else:
        wrapped_pos_frac = lattice.get_fractional_coords(wrapped_positions)

        return torch.tensor(wrapped_pos_frac, dtype=torch.float32)


def get_osda_mean_pbc(
    osda_coords_frac: torch.Tensor,
    lattice: torch.Tensor,
    edge_index: torch.Tensor,
    loading: int,
) -> torch.Tensor:
    """
    Function to calculate the mean position of osda with periodic boundary conditions.
    Return means of osda molecules in fractional coordinates.
    """
    osda_coords_cart = osda_coords_frac @ lattice.T

    # split pos into individual osda molecules
    split_osda_pos = torch.split(
        osda_coords_cart, osda_coords_cart.shape[0] // loading, dim=0
    )
    means = []

    # handle each molecule separately
    for osda_pos_i in split_osda_pos:
        # Wrap coordinates with PBC
        wrapped_coords_frac = fast_wrap_coords_edge_based(
            osda_pos_i, lattice, edge_index, return_cart=False
        )

        # get mean coordinate, and wrap it back to the unit cell
        mean = wrapped_coords_frac.mean(0) % 1.0
        means.append(mean.view(1, 3))

    means = torch.cat(means, dim=0)

    return means
