import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.utils import dense_to_sparse

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

    # Return the atomic numbers and coordinates as a tuple
    return torch.stack(atom_coords)


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
