import torch
from torch import nn
import math
from abc import ABC
from abc import abstractmethod

from rfm_docking.se3_dock.so3_utils import calc_rot_vf
from rfm_docking.se3_dock.so3_utils import expmap as so3_expmap
from rfm_docking.se3_dock.so3_utils import logmap as so3_logmap
from rfm_docking.se3_dock.so3_utils import geodesic_t as so3_geodesic

from rfm_docking.utils import sample_rotation_matrices


class SimpleManifold(ABC):
    @abstractmethod
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def geodesic(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        pass

    def get_u(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.logmap(x0, x1)

    @staticmethod
    @abstractmethod
    def projx(x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def random(*size, dtype=None, device=None) -> torch.Tensor:
        pass


class FlatTorus01(SimpleManifold):
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.projx(x + u)

    @staticmethod
    def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = 2 * math.pi * (y - x)
        return torch.atan2(torch.sin(z), torch.cos(z)) / (2 * math.pi)

    def geodesic(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        out = self.expmap(x0, t * self.logmap(x0, x1))
        return out

    @staticmethod
    def projx(x: torch.Tensor) -> torch.Tensor:
        return x % 1.0

    @staticmethod
    def random(*size, dtype=None, device=None) -> torch.Tensor:
        return torch.rand(*size, dtype=dtype, device=device)


class SO3(SimpleManifold):
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        out = so3_expmap(x, u)
        return out

    @staticmethod
    def logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = so3_logmap(x, y)
        return out

    @staticmethod
    def geodesic(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = so3_geodesic(x0, x1, t)
        return out

    def get_u(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return calc_rot_vf(x0, x1)

    @staticmethod
    def projx(x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def random(*size, dtype=None, device=None) -> torch.Tensor:
        return sample_rotation_matrices(size[0])


class SE3ProductManifold(nn.Module):
    """
    A class for the SE3 product manifold. Treat fractional coordinates and rotation separately.
    Inputs to all functions are batched, i.e. speedups with vmap are possible. We thus do *not* need
    masking.
    """

    def __init__(self):
        super().__init__()

        self.flat_torus = FlatTorus01()
        self.so3 = SO3()

    def forward(
        self,
        f0: torch.Tensor,
        f1: torch.Tensor,
        rotmat_0: torch.Tensor,
        rotmat_1: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        f0, f1: [B, 3]
        rotmat_0, rotmat_1: [B, 3, 3]
        t: [B, 1]

        Returns:
        f_t: [B, 3]
        rotmat_t: [B, 3, 3]
        u_f: [B, 3]
        u_rot: [B, 3]
        """

        f_t, rotmat_t = self.geodesic(f0, f1, rotmat_0, rotmat_1, t)

        # TODO /(1-t) or take logmap_f0(f1)
        # u_f, u_rot = self.get_u(f_t, f1, rotmat_t, rotmat_1, t)
        u_f, u_rot = self.get_u(f0, f1, rotmat_0, rotmat_1, t)

        return f_t, rotmat_t, u_f, u_rot

    def expmap(
        self, f: torch.Tensor, u_f: torch.Tensor, rot: torch.Tensor, u_rot
    ) -> torch.Tensor:
        """
        f, u_f: [B, 3]
        rot: [B, 3, 3]
        u_rot: [B, 3]

        Returns:
        f_t: [B, 3]
        rotmat_t: [B, 3, 3]
        """
        f_t = self.flat_torus.expmap(f, u_f)
        rotmat_t = self.so3.expmap(rot, u_rot)

        return f_t, rotmat_t

    def logmap(
        self,
        f: torch.Tensor,
        f1: torch.Tensor,
        rotmat: torch.Tensor,
        rotmat_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        f, f1: [B, 3]
        rotmat, rotmat_1: [B, 3, 3]

        Returns:
        log_f: [B, 3]
        log_rot: [B, 3]
        """
        log_f = self.flat_torus.logmap(f, f1)
        log_rot = self.so3.logmap(rotmat, rotmat_1)

        return log_f, log_rot

    def projx(self, f: torch.Tensor, rotmat: torch.Tensor) -> torch.Tensor:
        """
        f: [B, 3]
        rotmat: [B, 3, 3]

        Returns:
        proj_f: [B, 3]
        proj_rot: [B, 3, 3]
        """
        proj_f = self.flat_torus.projx(f)
        proj_rot = self.so3.projx(rotmat)

        return proj_f, proj_rot

    def geodesic(
        self,
        f0: torch.Tensor,
        f1: torch.Tensor,
        rotmat_0: torch.Tensor,
        rotmat_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        f0, f1: [B, 3]
        rotmat0, rotmat1: [B, 3, 3]
        t: [B, 1]

        Returns:
        f_t: [B, 3]
        rotmat_t: [B, 3, 3]
        """
        f_t = self.flat_torus.geodesic(f0, f1, t)
        rotmat_t = self.so3.geodesic(rotmat_0, rotmat_1, t)

        return f_t, rotmat_t

    def get_u(
        self,
        ft: torch.Tensor,
        f1: torch.Tensor,
        rotmat_t: torch.Tensor,
        rotmat_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        f0, f1: [B, 3]
        rotmat0, rotmat1: [B, 3, 3]

        Returns:
        u_f: [B, 3]
        u_rot: [B, 3]
        """
        # Calculate the tangent vectors (essentially the logmap)
        u_f = self.flat_torus.get_u(ft, f1)
        u_rot = self.so3.get_u(rotmat_t, rotmat_1)

        # Normalize the tangent vectors. Add a small epsilon to avoid division by zero.
        # u_f = u_f / (1 - t + 1e-8)
        # u_rot = u_rot / (1 - t + 1e-8)
        return u_f, u_rot


if __name__ == "__main__":
    se3_manifold = SE3ProductManifold()

    f0 = torch.rand(10, 3)
    f1 = torch.rand(10, 3)

    rotmat_0 = sample_rotation_matrices(10)
    rotmat_1 = sample_rotation_matrices(10)

    t = torch.rand(10, 1)

    f_t, rotmat_t, u_f, u_rot = se3_manifold(f0, f1, rotmat_0, rotmat_1, t)
    print(f_t.shape, rotmat_t.shape)

    u_f, u_rot = se3_manifold.get_u(f_t, f1, rotmat_t, rotmat_1, t)
    print(u_f.shape, u_rot.shape)
