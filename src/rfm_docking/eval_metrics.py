import torch
from scipy.optimize import linear_sum_assignment

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds

from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.utils import fast_wrap_coords_edge_based
from rfm_docking.featurization import get_bond_edges
from diffcsp.script_utils import chemical_symbols

def kabsch_torch(P, Q):
    """
    From https://hunterheidenreich.com/posts/kabsch_algorithm/
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

    return R, t, rmsd

def compare_distance_matrix(osda_pred, osda_target, loading, lattice):
    # split the osda_pred and osda_target with loading
    osda_preds_split = torch.split(osda_pred, osda_pred.shape[0] // loading)

    pred_d_vectors = [
        FlatTorus01.logmap(p[:, None, :], p[None, :, :]) for p in osda_preds_split
    ]
    pred_d_vectors = torch.stack(pred_d_vectors)

    # frac to cartesian
    pred_d_vectors = torch.matmul(pred_d_vectors, lattice.T)
    pred_d_matrices = (pred_d_vectors**2).sum(-1).sqrt()

    osda_targets_split = torch.split(osda_target, osda_target.shape[0] // loading)

    osda_target_d_matrices = [
        FlatTorus01.logmap(t[:, None, :], t[None, :, :]) for t in osda_targets_split
    ]
    osda_target_d_vectors = torch.stack(osda_target_d_matrices)

    # frac to cartesian
    osda_target_d_vectors = torch.matmul(osda_target_d_vectors, lattice.T)
    osda_target_d_matrices = (osda_target_d_vectors**2).sum(-1).sqrt()

    cost_vectors = pred_d_matrices[None, :, :] - osda_target_d_matrices[:, None, :]

    # take the norm in the last two dimensions
    cost_matrix = torch.norm(cost_vectors, dim=(-2, -1))
    cost_matrix = cost_matrix.numpy()

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # displacement on molecule level
    total_displacement = cost_matrix[row_ind, col_ind].sum()
    mean_mol_displacement = cost_matrix[row_ind, col_ind].mean()
    max_mol_displacement = cost_matrix[row_ind, col_ind].max()
    min_mol_displacement = cost_matrix[row_ind, col_ind].min()

    # displacement on atom level
    cost = torch.abs(pred_d_matrices - osda_target_d_matrices[col_ind])
    # remove diagonal
    diag_indices = torch.arange(cost.shape[-1])
    diag_mask = torch.ones_like(cost, dtype=bool)
    diag_mask[:, diag_indices, diag_indices] = 0

    mean_atom_displacement = cost[diag_mask].mean().item()
    max_atom_displacement = cost[diag_mask].max().item()
    min_atom_displacement = cost[diag_mask].min().item()

    displacements = {
        "total_displacement": total_displacement,
        "mean_total_displacement": mean_mol_displacement,
        "max_total_displacement": max_mol_displacement,
        "min_total_displacement": min_mol_displacement,
        "mean_atom_displacement": mean_atom_displacement,
        "max_atom_displacement": max_atom_displacement,
        "min_atom_displacement": min_atom_displacement,
    }

    return displacements

def check_rmsd(osda_pred_frac, osda_target_frac, lattice, loading, smiles):

    osda_target_cart = torch.matmul(osda_target_frac, lattice.T)
    osda_target_split = torch.split(osda_target_cart, osda_target_cart.shape[0] // loading)
    osda_pred_cart = torch.matmul(osda_pred_frac, lattice.T)
    osda_pred_split = torch.split(osda_pred_cart, osda_pred_cart.shape[0] // loading)

    # get edges from smiles
    mol_edges, _ = get_bond_edges(Chem.MolFromSmiles(smiles))

    pred = [fast_wrap_coords_edge_based(i, lattice, mol_edges, return_cart=True) for i in osda_pred_split]
    target = [fast_wrap_coords_edge_based(i, lattice, mol_edges, return_cart=True) for i in osda_target_split]

    min_rmsds = []
    for idx_i, i in enumerate(pred):
        rmsds = []
        for idx_j, j in enumerate(target):
            R, t, rmsd = kabsch_torch(i, j)
            rmsds.append(rmsd)
        min_rmsd = min(rmsds)
        min_rmsds.append(min_rmsd)


    rmsd = torch.tensor(min_rmsds).mean().cpu().item()
    return rmsd

# TODO atom is clashed once--> clash. do not count multiple times
def check_zeolite_osda_clash(zeolite_pos, osda_pos, loading, lattice, distance_cutoff=2.0):
    """
    Check clashes between osda and zeolite. Distance cutoff 1.5A taken from VOID + 1.1A = 2.6A for C-H bond.
    """
    distance_vectors = FlatTorus01.logmap(zeolite_pos[:, None, :], osda_pos[None, :, :])
    distance_vectors = torch.matmul(distance_vectors, lattice.T)
    distance_matrix = (distance_vectors**2).sum(-1).sqrt()

    is_clash = distance_matrix < distance_cutoff

    is_clash_atomwise = torch.any(is_clash, dim=-1)
    
    is_clash_split = torch.split(is_clash, is_clash.shape[0] // loading)
    is_clash_molwise = [torch.any(i).to(torch.int32).item() for i in is_clash_split]
    num_clash_molwise = sum(is_clash_molwise)

    num_clashes_atom = (is_clash_atomwise).sum().item() 

    return num_clash_molwise


def check_osda_osda_clash(osda_pos_frac, loading, lattice, distance_cutoff=2.2):
    """distance cutoff 2 * 1.1A = 2.2A for C-H bond."""
    if loading==1:
        # no clash possible
        return None
    atoms_per_mol = osda_pos_frac.shape[0] // loading

    distance_vector_frac = FlatTorus01.logmap(
        osda_pos_frac[:, None, :], osda_pos_frac[None, :, :]
    )

    # frac to cartesian
    distance_vector_cart = torch.matmul(distance_vector_frac, lattice.T)
    distance_matrix = (distance_vector_cart**2).sum(-1).sqrt()

    mask = torch.ones_like(distance_matrix, dtype=bool)
    mask = mask.triu(diagonal=1)

    # mask out intramolecular distances
    for i in range(loading):
        start = i * atoms_per_mol
        end = (i + 1) * atoms_per_mol
        mask[start:end, start:end] = False

    is_clash = distance_matrix < distance_cutoff
    is_clash = torch.logical_and(is_clash, mask)
    is_clash = torch.any(is_clash, dim=-1)

    num_clashes_atomwise = (is_clash).sum().item()

    is_clash_split = torch.split(is_clash, is_clash.shape[0] // loading)
    num_clashes_molwise = [torch.any(i).to(torch.int32).item() for i in is_clash_split]
    num_clashes_molwise = sum(num_clashes_molwise)
  
    return num_clashes_molwise

def get_average_path_length(com_traj_frac, osda_traj_frac, loading, lattice):
    # calculate path length based on frac coordinates
    len_traj = len(osda_traj_frac)

    path_len_com = 0
    if com_traj_frac is not None:
        path_len_cart = torch.zeros((com_traj_frac[0].shape[0], ))
        for i in range(1, len_traj):
            path_vector_i = FlatTorus01.logmap(com_traj_frac[i - 1], com_traj_frac[i])
            path_vector_cart = torch.matmul(path_vector_i.to(device=lattice.device), lattice.T)
            len_path_vector = (path_vector_cart**2).sum(-1).sqrt()
            path_len_cart += len_path_vector
        path_len_com = path_len_cart.mean().cpu().item()
    
    path_len_cart = torch.zeros((osda_traj_frac[0].shape[0], ))
    for i in range(1, len_traj):
        path_vector_i = FlatTorus01.logmap(osda_traj_frac[i - 1], osda_traj_frac[i])
        path_vector_cart = torch.matmul(path_vector_i.to(device=lattice.device), lattice.T)
        len_path_vector = (path_vector_cart**2).sum(-1).sqrt()
        path_len_cart += len_path_vector

    path_len_osda = path_len_cart.mean().cpu().item()

    total_path_len = path_len_com + path_len_osda
    return total_path_len
            



if __name__ == "__main__":
    from diffcsp.common.data_utils import (lattice_params_to_matrix_torch)
    fractional_coords = torch.tensor([
        [0.34844, 0.04845, 0.54481],
        [0.33462, 0.12540, 0.43489],
        [0.39142, 0.13989, 0.36699],
        [0.29502, 0.23509, 0.44972],
        [0.29825, 0.05266, 0.37213],
        [0.95331, 0.55835, 0.95848],
        [0.96315, 0.59393, 0.84639],
        [0.98064, 0.71439, 0.88162],
        [0.92329, 0.66011, 0.75672],
        [0.96819, 0.47061, 0.79152]
    ])
    lattice = torch.tensor([25.02820,  5.05502, 12.30120,  90.0001, 108.0410,  90.0000])
    lattice = lattice_params_to_matrix_torch(lattice[:3].unsqueeze(0), lattice[3:].unsqueeze(0)).squeeze()
    loading = 2

    clashes = check_osda_osda_clash(fractional_coords, loading, lattice)