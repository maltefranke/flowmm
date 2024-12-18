import torch
from scipy.optimize import linear_sum_assignment


from src.flowmm.rfm.manifolds.flat_torus import FlatTorus01


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

    mean_atom_displacement = cost[diag_mask].mean()
    max_atom_displacement = cost[diag_mask].max()
    min_atom_displacement = cost[diag_mask].min()

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


def check_zeolite_osda_clash(zeolite_pos, osda_pos, lattice, distance_cutoff=2.5):
    distance_vectors = FlatTorus01.logmap(zeolite_pos[:, None, :], osda_pos[None, :, :])
    distance_vectors = torch.matmul(distance_vectors, lattice.T)
    distance_matrix = (distance_vectors**2).sum(-1).sqrt()

    # divide by 2 because the distance matrix is symmetric
    num_clashes = (distance_matrix < distance_cutoff).sum().item() / 2

    min_distance = distance_matrix.min().item()
    max_distance = distance_matrix.max().item()
    mean_distance = distance_matrix.mean().item()

    clash_stats = {
        "num_clashes": num_clashes,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "mean_distance": mean_distance,
    }

    return clash_stats


def check_osda_osda_clash(osda_pos_frac, loading, lattice, distance_cutoff=2.5):
    atoms_per_mol = osda_pos_frac.shape[0] // loading

    distance_vector_frac = FlatTorus01.logmap(
        osda_pos_frac[:, None, :], osda_pos_frac[None, :, :]
    )

    # frac to cartesian
    distance_vector_cart = torch.matmul(distance_vector_frac, lattice.T)
    distance_matrix = (distance_vector_cart**2).sum(-1).sqrt()

    mask = torch.ones_like(distance_matrix, dtype=bool)
    # mask out intramolecular distances
    for i in range(loading):
        start = i * atoms_per_mol
        end = (i + 1) * atoms_per_mol
        mask[start:end, start:end] = False

    intermolecular_distances = distance_matrix[mask]

    # divide by 2 because the distance matrix is symmetric
    num_clashes = (intermolecular_distances < distance_cutoff).sum().item() / 2

    min_distance = intermolecular_distances.min().item()
    max_distance = intermolecular_distances.max().item()
    mean_distance = intermolecular_distances.mean().item()

    clash_stats = {
        "num_clashes": num_clashes,
        "min_distance": min_distance,
        "max_distance": max_distance,
        "mean_distance": mean_distance,
    }

    return clash_stats


if __name__ == "__main__":
    osda_pred = torch.rand(20, 3)
    # osda_target = torch.rand(20, 3)
    osda_target = (osda_pred + 0.1) % 1.0
    loading = 4
    lattice = torch.eye(3)
    # cost = compare_distance_matrix(osda_pred, osda_target, loading)
    cost = check_osda_osda_clash(osda_pred, loading, lattice)
    print(cost)
