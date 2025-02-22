import torch
from geoopt import Euclidean, Sphere

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


class NSphere(Sphere):
    """
    N-spheres, one for each molecule
    """

    def __init__(self, num_mols: int, max_num_mols: int):
        super().__init__()
        self.max_num_mols = max_num_mols
        mask = torch.zeros(max_num_mols, dtype=torch.bool)
        mask[:num_mols] = torch.ones(num_mols, dtype=torch.bool)
        self.register_buffer("mask", mask.unsqueeze(-1))

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape

        out = super().expmap(x.view(-1, 3), u.view(-1, 3))
        out = torch.nan_to_num(out, nan=0.0)
        out = out.reshape(x_shape)
        return out

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape

        out = super().logmap(x.view(-1, 3), y.view(-1, 3))
        out = torch.nan_to_num(out, nan=0.0)
        out = out.reshape(x_shape)
        return out

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape

        out = super().projx(x.view(-1, 3))
        out = torch.nan_to_num(out, nan=0.0)
        out = out.reshape(x_shape)
        return out

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape

        out = super().proju(x, u)
        out = torch.nan_to_num(out, nan=0.0)
        out = out.reshape(x_shape)
        return out
