import torch
from geoopt import Euclidean

from flowmm.rfm.manifolds.flat_torus import MaskedNoDriftFlatTorus01


class DockingFlatTorus01(MaskedNoDriftFlatTorus01):
    """Represents a flat torus on the [0, 1]^D subspace.

    Isometric to the product of 1-D spheres.
    Leaves target vector field translation equivariant.
    To be used when only flowing the OSDA atoms (but not the zeolite atoms).
    """

    name = "DockingFlatTorus01"
    reversible = False

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Does not remove the drift velocity, therefore the u_t is translation equivariant.
        """
        initial_shape = u.shape
        out = self.reshape_and_mask(initial_shape, u)

        out = self.mask_and_reshape(initial_shape, out)
        return super(MaskedNoDriftFlatTorus01, self).proju(x, out)


class OptimizeFlatTorus01(DockingFlatTorus01):
    """Represents a flat torus on the [0, 1]^D subspace.

    Isometric to the product of 1-D spheres.
    Target vector field will be translation invariant. To be used when flowing all atoms.
    """

    name = "OptimizeFlatTorus01"
    reversible = False
