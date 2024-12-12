import torch
import numpy as np
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
from pymatgen.core import Structure, Lattice

from flowmm.rfm.manifolds.flat_torus import FlatTorus01


def get_voronoi_nodes(
    zeolite_pos: np.array, lattice: np.array, cartesian: bool = True, cutoff: float = 3
) -> np.array:
    # Compute Voronoi diagram
    vor = Voronoi(zeolite_pos)

    # Get Voronoi nodes
    voronoi_pos = vor.vertices

    # cartesian to fractional
    zeolite_pos_frac = np.dot(zeolite_pos, np.linalg.inv(lattice))
    voronoi_pos_frac = np.dot(voronoi_pos, np.linalg.inv(lattice))

    # remove voronoi nodes outside the unit cell -> they must not be wrapped back in
    is_outside = np.any(voronoi_pos_frac > 1, axis=1) | np.any(
        voronoi_pos_frac < 0, axis=1
    )
    voronoi_pos = voronoi_pos[~is_outside]
    voronoi_pos_frac = voronoi_pos_frac[~is_outside]

    # to torch needed for log map
    zeolite_pos_frac_torch = torch.tensor(zeolite_pos_frac)
    voronoi_pos_frac_torch = torch.tensor(voronoi_pos_frac)

    distance_vectors = FlatTorus01.logmap(
        zeolite_pos_frac_torch[:, None, :], voronoi_pos_frac_torch[None, :, :]
    ).numpy()

    # frac to cartesian
    distance_vectors = np.dot(distance_vectors, lattice)

    distance_tensor = np.sqrt(
        (distance_vectors**2).sum(
            -1,
        )
    )

    # Find the indices of voronoi nodes that are closer than 3 to at least one zeolite position
    mask = np.any(distance_tensor < cutoff, axis=0)

    # Mask out the voronoi nodes within the cutoff
    if cartesian:
        return voronoi_pos[~mask]
    else:
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
