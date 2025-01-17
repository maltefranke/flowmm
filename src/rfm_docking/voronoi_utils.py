import torch
import numpy as np
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
from pymatgen.core import Structure, Lattice

from flowmm.rfm.manifolds.flat_torus import FlatTorus01


def get_voronoi_nodes(
    zeolite_pos_frac: torch.Tensor, lattice: np.array, cutoff: float = 3
) -> np.array:
    structure = Structure(
        Lattice(lattice), ["H"] * len(zeolite_pos_frac), zeolite_pos_frac
    )

    # make 333 supercell
    supercell = structure.make_supercell([3, 3, 3])

    supercell_pos_frac = supercell.frac_coords

    # Compute Voronoi diagram
    vor = Voronoi(supercell_pos_frac)

    # Get Voronoi nodes
    voronoi_pos_frac = vor.vertices

    # put 111 cell between 0 and 1/3
    voronoi_pos_frac -= 1 / 3
    # put 111 cell between 0 and 1
    voronoi_pos_frac *= 3

    # remove voronoi nodes outside the unit cell -> they must not be wrapped back in
    is_outside = np.any(voronoi_pos_frac > 1, axis=1) | np.any(
        voronoi_pos_frac < 0, axis=1
    )

    voronoi_pos_frac = voronoi_pos_frac[~is_outside]

    # to torch needed for log map
    voronoi_pos_frac_torch = torch.tensor(voronoi_pos_frac)

    distance_vectors = FlatTorus01.logmap(
        zeolite_pos_frac[:, None, :], voronoi_pos_frac_torch[None, :, :]
    ).numpy()

    # frac to cartesian
    distance_vectors = np.dot(distance_vectors, lattice)

    distance_tensor = np.sqrt(
        (distance_vectors**2).sum(
            -1,
        )
    )

    # Find the indices of voronoi nodes that are closer than 3A to at least one zeolite position
    mask = np.any(distance_tensor < cutoff, axis=0)

    if voronoi_pos_frac[~mask].shape[0] == 0:
        print("distance_tensor: ", distance_tensor)
        print("zeolite_pos_frac: ", zeolite_pos_frac)
        print("voronoi_pos_frac: ", voronoi_pos_frac)
        print("mask: ", mask)

    # Mask out the voronoi nodes within the cutoff
    return voronoi_pos_frac[~mask]


def cluster_voronoi_nodes(
    voronoi_pos_frac: np.array,
    lattice_np: np.array,
    cutoff: float = 13.0,
    merge_tol: float = 2.0,
) -> torch.Tensor:
    lattice = Lattice(lattice_np)

    clustering = DBSCAN(eps=cutoff, min_samples=2, metric="precomputed")

    voronoi_pos_frac_torch = torch.tensor(voronoi_pos_frac)

    # distance matrix in fractional coordinates
    distance_vectors = FlatTorus01.logmap(
        voronoi_pos_frac_torch[:, None, :], voronoi_pos_frac_torch[None, :, :]
    ).numpy()

    # frac to cartesian
    distance_vectors = np.dot(distance_vectors, lattice_np)

    # cartesian distance matrix
    distance_tensor = np.sqrt(
        (distance_vectors**2).sum(
            -1,
        )
    )

    voronoi_classes = clustering.fit_predict(distance_tensor)

    merged_voronoi_pos = []
    # use pymatgen merge_sites to merge the voronoi nodes
    for voronoi_class in np.unique(voronoi_classes):
        voronoi_pos_i = voronoi_pos_frac[voronoi_classes == voronoi_class]

        # set H as dummy atoms
        struct = Structure(lattice, ["H"] * len(voronoi_pos_i), voronoi_pos_i)

        merged_voronoi_pos_i = struct.merge_sites(mode="average", tol=merge_tol)

        merged_voronoi_pos.append(
            torch.tensor(merged_voronoi_pos_i.frac_coords, dtype=torch.float32)
        )

    merged_voronoi_pos = torch.cat(merged_voronoi_pos)

    return merged_voronoi_pos


if __name__ == "__main__":
    zeolite_pos_frac = torch.tensor(
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )

    lattice = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])

    voronoi_pos_frac = get_voronoi_nodes(zeolite_pos_frac, lattice)

    merged_voronoi_pos = cluster_voronoi_nodes(voronoi_pos_frac, lattice)
    print(merged_voronoi_pos)
