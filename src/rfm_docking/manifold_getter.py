"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

from collections import namedtuple
from typing import Literal

import numpy as np
import torch
from torch_geometric.utils import to_dense_batch

from flowmm.cfg_utils import dataset_options
from flowmm.geometric_ import mask_2d_to_batch
from flowmm.rfm.manifolds import (
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
    ProductManifoldWithLogProb,
)
from flowmm.rfm.manifolds.flat_torus import (
    MaskedNoDriftFlatTorus01,
    MaskedNoDriftFlatTorus01WrappedNormal,
)
from rfm_docking.manifolds import DockingFlatTorus01, OptimizeFlatTorus01, NSphere
from flowmm.rfm.vmap import VMapManifolds
from geoopt import Sphere

Dims = namedtuple("Dims", ["f"])
ManifoldGetterOut = namedtuple(
    "ManifoldGetterOut", ["flat", "manifold", "dims", "mask_f"]
)
SplitManifoldGetterOut = namedtuple(
    "SplitManifoldGetterOut",
    [
        "flat",
        "manifold",
        "f_manifold",
        "dims",
        "mask_f",
    ],
)
GeomTuple = namedtuple("GeomTuple", ["f"])
coord_manifold_types = Literal[
    "flat_torus_01",
    "flat_torus_01_normal",
    "flat_torus_01_fixfirst",
    "flat_torus_01_fixfirst_normal",
    "docking_flat_torus_01",
    "optimize_flat_torus_01",
]


class DockingManifoldGetter(torch.nn.Module):
    """Only contains the coord manifold"""

    def __init__(
        self,
        coord_manifold: coord_manifold_types,
        dataset: dataset_options | None = None,
    ) -> None:
        super().__init__()
        self.coord_manifold = coord_manifold
        self.dataset = dataset

    @staticmethod
    def _get_max_num_atoms(mask_f: torch.BoolTensor) -> int:
        """Returns the maximum number of atoms in the batch"""
        return int(mask_f.sum(dim=-1).max())

    @staticmethod
    def _get_num_atoms(mask_f: torch.BoolTensor) -> torch.LongTensor:
        """Returns the number of atoms in each graph in the batch"""
        return mask_f.sum(dim=-1)

    def _to_dense(
        self,
        batch: torch.LongTensor,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Converts sparse batches of variable-sized graphs into dense tensors.
        Pads each graph in the batch with zeros so that all graphs have the same number of nodes.
        Returns a mask tensor that indicates the valid nodes for each graph.
        """
        f, mask_f = to_dense_batch(x=frac_coords, batch=batch)  # B x N x 3, B x N
        return f, mask_f

    def _to_flat(
        self,
        f: torch.Tensor,
        dims: Dims,
    ) -> torch.Tensor:
        f_flat = f.reshape(f.size(0), dims.f)
        return f_flat

    def georep_to_flatrep(
        self,
        batch: torch.LongTensor,
        frac_coords: torch.Tensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts from georep to the manifold flatrep"""
        f, mask_f = self._to_dense(batch, frac_coords=frac_coords)
        num_atoms = self._get_num_atoms(mask_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms,
            dim_coords=frac_coords.shape[-1],
            split_manifold=split_manifold,
        )
        # manifolds = [manifold.to(device=batch.device) for manifold in manifolds]
        flat = self._to_flat(f, dims)
        if split_manifold:
            return SplitManifoldGetterOut(
                flat,
                *manifolds,
                dims,
                mask_f,
            )
        else:
            return ManifoldGetterOut(
                flat,
                *manifolds,
                dims,
                mask_f,
            )

    def forward(
        self,
        batch: torch.LongTensor,
        frac_coords: torch.Tensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """Converts data from the loader into the georep, then to the manifold flatrep"""
        return self.georep_to_flatrep(batch, frac_coords, split_manifold)

    def from_empty_batch(
        self,
        batch: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[
            tuple[int, ...],
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """Builds the manifolds from an empty batch"""
        _, mask_f = to_dense_batch(
            x=torch.zeros((*batch.shape, 1), device=batch.device), batch=batch
        )  # B x N
        num_atoms = self._get_num_atoms(mask_f)
        *manifolds, dims = self.get_manifolds(
            num_atoms, dim_coords, split_manifold=split_manifold
        )
        return (len(num_atoms), sum(dims)), *manifolds, dims, mask_f

    @staticmethod
    def _from_dense(
        f: torch.Tensor,
        mask_f: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms the dense representation into its sparse representation"""
        return f[mask_f]

    def _from_flat(
        self,
        flat: torch.Tensor,
        dims: Dims,
        max_num_atoms: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, _ = torch.tensor_split(flat, np.cumsum(dims).tolist(), dim=1)

        return f.reshape(-1, max_num_atoms, dims.f // max_num_atoms)

    def flatrep_to_georep(
        self, flat: torch.Tensor, dims: Dims, mask_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to the georep"""
        max_num_atoms = self._get_max_num_atoms(mask_f)
        f = self._from_flat(flat, dims, max_num_atoms)
        f = self._from_dense(f, mask_f)
        return GeomTuple(f)

    def georep_to_crystal(
        self,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the georep to (one_hot / bits, frac_coords, lattice matrix)"""

        return frac_coords

    def flatrep_to_crystal(
        self, flat: torch.Tensor, dims: Dims, mask_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to (one_hot / bits, frac_coords, lattice matrix)"""
        return self.georep_to_crystal(*self.flatrep_to_georep(flat, dims, mask_f))

    @staticmethod
    def mask_f_to_batch(mask_f: torch.BoolTensor) -> torch.LongTensor:
        return mask_2d_to_batch(mask_f)

    @staticmethod
    def _get_manifold(
        num_atom: int,
        dim_coords: int,
        max_num_atoms: int,
        split_manifold: bool,
        coord_manifold: coord_manifold_types,
    ) -> (
        tuple[ProductManifoldWithLogProb]
        | FlatTorus01FixFirstAtomToOrigin
        | FlatTorus01FixFirstAtomToOriginWrappedNormal
        | MaskedNoDriftFlatTorus01
        | MaskedNoDriftFlatTorus01WrappedNormal,
    ):  # type: ignore
        if coord_manifold == "flat_torus_01":
            f_manifold = (
                MaskedNoDriftFlatTorus01(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_normal":
            f_manifold = (
                MaskedNoDriftFlatTorus01WrappedNormal(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "docking_flat_torus_01":
            f_manifold = (
                DockingFlatTorus01(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "optimize_flat_torus_01":
            f_manifold = (
                OptimizeFlatTorus01(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_fixfirst":
            f_manifold = (
                FlatTorus01FixFirstAtomToOrigin(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        elif coord_manifold == "flat_torus_01_fixfirst_normal":
            f_manifold = (
                FlatTorus01FixFirstAtomToOriginWrappedNormal(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        else:
            raise ValueError(f"{coord_manifold=} not in {coord_manifold_types=}")

        manifolds = [f_manifold]
        if split_manifold:
            return (
                ProductManifoldWithLogProb(*manifolds),
                f_manifold[0],
            )
        else:
            return ProductManifoldWithLogProb(*manifolds)

    def get_dims(self, dim_coords: int, max_num_atoms: int) -> Dims:
        dim_f = dim_coords * max_num_atoms

        return Dims(dim_f)

    def get_manifolds(
        self,
        num_atoms: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[VMapManifolds, tuple[int]]
        | tuple[
            VMapManifolds,
            VMapManifolds,
            tuple[int],
        ]
    ):
        max_num_atoms = num_atoms.amax(0).cpu().item()

        out_manifolds, f_manifolds = [], []
        for batch_idx, num_atom in enumerate(num_atoms):
            manis = self._get_manifold(
                num_atom,
                dim_coords,
                max_num_atoms,
                split_manifold=split_manifold,
                coord_manifold=self.coord_manifold,
            )
            if split_manifold:
                out_manifolds.append(manis[0])
                f_manifolds.append(manis[1])
            else:
                out_manifolds.append(manis)

        if split_manifold:
            return (
                VMapManifolds(out_manifolds),
                VMapManifolds(f_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )
        else:
            return (
                VMapManifolds(out_manifolds),
                self.get_dims(dim_coords, max_num_atoms),
            )


coord_manifold_types = Literal["docking_flat_torus_01",]
rot_manifold_types = Literal["sphere",]
tor_manifold_types = Literal["docking_flat_torus_01",]

SE3Dims = namedtuple("Dims", ["f", "rot"])
SE3ManifoldGetterOut = namedtuple(
    "ManifoldGetterOut", ["flat", "manifold", "dims", "mask_f"]
)
SE3SplitManifoldGetterOut = namedtuple(
    "SplitManifoldGetterOut",
    [
        "flat",
        "manifold",
        "f_manifold",
        "rot_manifold",
        "dims",
        "mask_f",
    ],
)
SE3GeomTuple = namedtuple("GeomTuple", ["f", "rot"])


class SE3ManifoldGetter(torch.nn.Module):
    def __init__(
        self,
        coord_manifold: coord_manifold_types,
        rot_manifold: rot_manifold_types,
        dataset: dataset_options | None = None,
    ) -> None:
        super().__init__()
        self.coord_manifold = coord_manifold
        self.rot_manifold = rot_manifold
        self.dataset = dataset

    @staticmethod
    def _get_max_num_atoms(mask: torch.BoolTensor) -> int:
        return int(mask.sum(dim=-1).max())

    @staticmethod
    def _get_num_atoms(mask_a_or_f: torch.BoolTensor) -> torch.LongTensor:
        return mask_a_or_f.sum(dim=-1)

    def _to_dense(
        self,
        coord_batch: torch.LongTensor,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        f, mask_f = to_dense_batch(x=frac_coords, batch=coord_batch)  # B x N x 3
        return (
            f,
            mask_f,
        )

    def _to_flat(
        self,
        f: torch.Tensor,
        rot: torch.Tensor,
        dims: SE3Dims,
    ) -> torch.Tensor:
        f_flat = f.reshape(f.size(0), dims.f)
        rot_flat = rot.reshape(rot.size(0), dims.rot)
        return torch.cat([f_flat, rot_flat], dim=1)

    def georep_to_flatrep(
        self,
        frac_coords: torch.Tensor,
        rot: torch.Tensor,
        coord_batch: torch.LongTensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts from georep to the manifold flatrep"""
        f, mask_f = self._to_dense(coord_batch, frac_coords)
        rot, _ = self._to_dense(coord_batch, rot)
        num_mols = mask_f.sum(dim=-1)
        *manifolds, dims = self.get_manifolds(
            num_mols,
            dim_coords=frac_coords.shape[-1],
            split_manifold=split_manifold,
        )
        flat = self._to_flat(f, rot, dims)
        if split_manifold:
            return SE3SplitManifoldGetterOut(flat, *manifolds, dims, mask_f)
        else:
            return SE3ManifoldGetterOut(flat, *manifolds, dims, mask_f)

    def forward(
        self,
        frac_coords,
        rot,
        coord_batch,
        split_manifold,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts data from the loader into the georep, then to the manifold flatrep"""

        return self.georep_to_flatrep(
            frac_coords,
            rot,
            coord_batch,
            split_manifold,
        )

    def from_empty_batch(
        self,
        coord_batch: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[
            tuple[int, ...],
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        _, mask_f = to_dense_batch(
            x=torch.zeros((*coord_batch.shape, 1), device=coord_batch.device),
            batch=coord_batch,
        )  # B x N
        num_mols = mask_f.sum(dim=-1)

        *manifolds, dims = self.get_manifolds(
            num_mols, dim_coords, split_manifold=split_manifold
        )
        return (len(num_mols), sum(dims)), *manifolds, dims, mask_f

    @staticmethod
    def _from_dense(
        f: torch.Tensor,
        mask_f: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return f[mask_f]

    def _from_flat(
        self,
        flat: torch.Tensor,
        dims: Dims,
        max_num_mols: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, rot, _ = torch.tensor_split(flat, np.cumsum(dims).tolist(), dim=1)
        f = f.reshape(-1, max_num_mols, dims.f // max_num_mols)
        rot = rot.reshape(-1, max_num_mols, dims.rot // max_num_mols)
        return f, rot

    def flatrep_to_georep(
        self,
        flat: torch.Tensor,
        dims: Dims,
        mask_f: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to the georep"""
        max_num_mols = self._get_max_num_atoms(mask_f)
        f, rot = self._from_flat(flat, dims, max_num_mols)
        f = self._from_dense(f, mask_f)
        rot = self._from_dense(rot, mask_f)
        return SE3GeomTuple(f, rot)

    def georep_to_crystal(
        self,
        frac_coords: torch.Tensor,
        rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """converts from the georep to (one_hot / bits, frac_coords, lattice matrix)"""

        return frac_coords, rot

    def flatrep_to_crystal(
        self, flat: torch.Tensor, dims: Dims, mask_a_or_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to (one_hot / bits, frac_coords, lattice matrix)"""
        return self.georep_to_crystal(*self.flatrep_to_georep(flat, dims, mask_a_or_f))

    @staticmethod
    def _get_manifold(
        num_atom: int,
        dim_coords: int,
        max_num_mols: int,
        split_manifold: bool,
        coord_manifold: coord_manifold_types,
        rot_manifold: rot_manifold_types,
    ):
        if coord_manifold == "docking_flat_torus_01":
            f_manifold = (
                DockingFlatTorus01(
                    dim_coords,
                    num_atom,
                    max_num_mols,
                ),
                dim_coords * max_num_mols,
            )
        else:
            raise ValueError(f"{coord_manifold=} not in {coord_manifold_types=}")

        if rot_manifold == "sphere":
            rot_manifold = (
                NSphere(num_atom, max_num_mols),
                dim_coords * max_num_mols,
            )
        else:
            raise ValueError(f"{rot_manifold=} not in {rot_manifold=}")

        manifolds = [f_manifold] + [rot_manifold]
        if split_manifold:
            return (
                ProductManifoldWithLogProb(*manifolds),
                f_manifold[0],
                rot_manifold[0],
            )
        else:
            return ProductManifoldWithLogProb(*manifolds)

    def get_dims(self, dim_coords: int, max_num_mols: int) -> Dims:
        dim_f = dim_coords * max_num_mols
        dim_rot = 3 * max_num_mols

        return SE3Dims(dim_f, dim_rot)

    def get_manifolds(
        self,
        num_mols: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[VMapManifolds, tuple[int, int, int]]
        | tuple[
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            tuple[int, int, int],
        ]
    ):
        max_num_mols = num_mols.amax(0).cpu().item()

        out_manifolds, f_manifolds, rot_manifolds = [], [], []
        for batch_idx, num_atom in enumerate(num_mols):
            manis = self._get_manifold(
                num_atom,
                dim_coords,
                max_num_mols,
                split_manifold=split_manifold,
                coord_manifold=self.coord_manifold,
                rot_manifold=self.rot_manifold,
            )
            if split_manifold:
                out_manifolds.append(manis[0])
                f_manifolds.append(manis[1])
                rot_manifolds.append(manis[2])
            else:
                out_manifolds.append(manis)

        if split_manifold:
            return (
                VMapManifolds(out_manifolds),
                VMapManifolds(f_manifolds),
                VMapManifolds(rot_manifolds),
                self.get_dims(dim_coords, max_num_mols),
            )
        else:
            return (
                VMapManifolds(out_manifolds),
                self.get_dims(dim_coords, max_num_mols),
            )


ProductSpaceDims = namedtuple("Dims", ["f", "rot", "tor"])
ProductSpaceManifoldGetterOut = namedtuple(
    "ManifoldGetterOut", ["flat", "manifold", "dims", "mask_f"]
)
ProductSpaceSplitManifoldGetterOut = namedtuple(
    "SplitManifoldGetterOut",
    [
        "flat",
        "manifold",
        "f_manifold",
        "rot_manifold",
        "tor_manifold",
        "dims",
        "mask_f",
        "mask_tor",
    ],
)
ProductSpaceGeomTuple = namedtuple("GeomTuple", ["f", "rot", "tor"])


class ProductSpaceManifoldGetter(torch.nn.Module):
    def __init__(
        self,
        coord_manifold: coord_manifold_types,
        rot_manifold: rot_manifold_types,
        tor_manifold: tor_manifold_types,
        dataset: dataset_options | None = None,
    ) -> None:
        super().__init__()
        self.coord_manifold = coord_manifold
        self.rot_manifold = rot_manifold
        self.tor_manifold = tor_manifold
        self.dataset = dataset

    @staticmethod
    def _get_max_num_atoms(mask: torch.BoolTensor) -> int:
        return int(mask.sum(dim=-1).max())

    @staticmethod
    def _get_num_atoms(mask_a_or_f: torch.BoolTensor) -> torch.LongTensor:
        return mask_a_or_f.sum(dim=-1)

    def _to_dense(
        self,
        coord_batch: torch.LongTensor,
        frac_coords: torch.Tensor,
        tor_batch: torch.LongTensor,
        tor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        f, mask_f = to_dense_batch(x=frac_coords, batch=coord_batch)  # B x N x 3
        tor, mask_tor = to_dense_batch(x=tor, batch=tor_batch)  # B x N x 3
        return f, mask_f, tor, mask_tor

    def _to_flat(
        self,
        f: torch.Tensor,
        rot: torch.Tensor,
        tor: torch.Tensor,
        dims: ProductSpaceDims,
    ) -> torch.Tensor:
        f_flat = f.reshape(f.size(0), dims.f)
        rot_flat = rot.reshape(rot.size(0), dims.rot)
        tor_flat = tor.reshape(tor.size(0), dims.tor)
        return torch.cat([f_flat, rot_flat, tor_flat], dim=1)

    def georep_to_flatrep(
        self,
        frac_coords: torch.Tensor,
        rot: torch.Tensor,
        tor: torch.Tensor,
        coord_batch: torch.LongTensor,
        tor_batch: torch.LongTensor,
        split_manifold: bool,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts from georep to the manifold flatrep"""
        f, mask_f, tor, mask_tor = self._to_dense(
            coord_batch, frac_coords, tor_batch, tor
        )
        num_atoms = mask_f.sum(dim=-1)
        num_rot_bonds = mask_tor.sum(dim=-1)
        *manifolds, dims = self.get_manifolds(
            num_atoms,
            num_rot_bonds,
            dim_coords=frac_coords.shape[-1],
            split_manifold=split_manifold,
        )
        flat = self._to_flat(f, rot, tor, dims)
        if split_manifold:
            return ProductSpaceSplitManifoldGetterOut(
                flat, *manifolds, dims, mask_f, mask_tor
            )
        else:
            return ProductSpaceManifoldGetterOut(
                flat, *manifolds, dims, mask_f, mask_tor
            )

    def forward(
        self,
        frac_coords,
        rot,
        tor,
        coord_batch,
        tor_batch,
        split_manifold,
    ) -> (
        tuple[
            torch.Tensor,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        """converts data from the loader into the georep, then to the manifold flatrep"""

        return self.georep_to_flatrep(
            frac_coords,
            rot,
            tor,
            coord_batch,
            tor_batch,
            split_manifold,
        )

    def from_empty_batch(
        self,
        coord_batch: torch.LongTensor,
        tor_batch: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[
            tuple[int, ...],
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            Dims,
            torch.BoolTensor,
        ]
        | tuple[tuple[int, ...], torch.Tensor, VMapManifolds, Dims, torch.BoolTensor]
    ):
        _, mask_f = to_dense_batch(
            x=torch.zeros((*coord_batch.shape, 1), device=coord_batch.device),
            batch=coord_batch,
        )  # B x N
        num_atoms = mask_f.sum(dim=-1)

        _, mask_tor = to_dense_batch(
            x=torch.zeros((*tor_batch.shape, 1), device=tor_batch.device),
            batch=tor_batch,
        )
        num_rot_bonds = mask_tor.sum(dim=-1)

        *manifolds, dims = self.get_manifolds(
            num_atoms, num_rot_bonds, dim_coords, split_manifold=split_manifold
        )
        return (len(num_atoms), sum(dims)), *manifolds, dims, mask_f, mask_tor

    @staticmethod
    def _from_dense(
        f: torch.Tensor,
        tor: torch.Tensor,
        mask_f: torch.BoolTensor,
        mask_tor: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return f[mask_f], tor[mask_tor]

    def _from_flat(
        self,
        flat: torch.Tensor,
        dims: Dims,
        max_num_atoms: int,
        max_num_rot_bonds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, rot, tor, _ = torch.tensor_split(flat, np.cumsum(dims).tolist(), dim=1)
        f = f.reshape(-1, max_num_atoms, dims.f // max_num_atoms)
        rot = rot.reshape(-1, dims.rot)
        tor = tor.reshape(-1, max_num_rot_bonds, dims.tor // max_num_rot_bonds)
        return f, rot, tor

    def flatrep_to_georep(
        self,
        flat: torch.Tensor,
        dims: Dims,
        mask_f: torch.BoolTensor,
        mask_tor: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to the georep"""
        max_num_atoms = self._get_max_num_atoms(mask_f)
        max_num_rot_bonds = self._get_max_num_atoms(mask_tor)
        f, rot, tor = self._from_flat(flat, dims, max_num_atoms, max_num_rot_bonds)
        f, tor = self._from_dense(f, tor, mask_f, mask_tor)
        return ProductSpaceGeomTuple(f, rot, tor)

    def georep_to_crystal(
        self,
        frac_coords: torch.Tensor,
        rot: torch.Tensor,
        tor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the georep to (one_hot / bits, frac_coords, lattice matrix)"""

        return frac_coords, rot, tor

    def flatrep_to_crystal(
        self, flat: torch.Tensor, dims: Dims, mask_a_or_f: torch.BoolTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """converts from the manifold flatrep to (one_hot / bits, frac_coords, lattice matrix)"""
        return self.georep_to_crystal(*self.flatrep_to_georep(flat, dims, mask_a_or_f))

    @staticmethod
    def _get_manifold(
        num_atom: int,
        dim_coords: int,
        max_num_atoms: int,
        split_manifold: bool,
        coord_manifold: coord_manifold_types,
        rot_manifold: rot_manifold_types,
        tor_manifold: tor_manifold_types,
    ):
        if coord_manifold == "docking_flat_torus_01":
            f_manifold = (
                DockingFlatTorus01(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        else:
            raise ValueError(f"{coord_manifold=} not in {coord_manifold_types=}")

        if rot_manifold == "sphere":
            rot_manifold = (Sphere(), 3)
        else:
            raise ValueError(f"{rot_manifold=} not in {rot_manifold=}")

        if tor_manifold == "docking_flat_torus_01":
            tor_manifold = (
                DockingFlatTorus01(
                    dim_coords,
                    num_atom,
                    max_num_atoms,
                ),
                dim_coords * max_num_atoms,
            )
        else:
            raise ValueError(f"{tor_manifold=} not in {tor_manifold_types=}")

        manifolds = [f_manifold] + [rot_manifold] + [tor_manifold]
        if split_manifold:
            return (
                ProductManifoldWithLogProb(*manifolds),
                f_manifold[0],
                rot_manifold[0],
                tor_manifold[0],
            )
        else:
            return ProductManifoldWithLogProb(*manifolds)

    def get_dims(
        self, dim_coords: int, max_num_atoms: int, max_num_rot_bonds: int
    ) -> Dims:
        dim_f = dim_coords * max_num_atoms
        dim_rot = 3

        dim_tor = dim_coords * max_num_rot_bonds

        return ProductSpaceDims(dim_f, dim_rot, dim_tor)

    def get_manifolds(
        self,
        num_atoms: torch.LongTensor,
        num_rot_bonds: torch.LongTensor,
        dim_coords: int,
        split_manifold: bool,
    ) -> (
        tuple[VMapManifolds, tuple[int, int, int]]
        | tuple[
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            VMapManifolds,
            tuple[int, int, int],
        ]
    ):
        max_num_atoms = num_atoms.amax(0).cpu().item()
        max_num_rot_bonds = num_rot_bonds.amax(0).cpu().item()

        out_manifolds, f_manifolds, rot_manifolds, tor_manifolds = [], [], [], []
        for batch_idx, num_atom in enumerate(num_atoms):
            manis = self._get_manifold(
                num_atom,
                dim_coords,
                max_num_atoms,
                split_manifold=split_manifold,
                coord_manifold=self.coord_manifold,
                rot_manifold=self.rot_manifold,
                tor_manifold=self.tor_manifold,
            )
            if split_manifold:
                out_manifolds.append(manis[0])
                f_manifolds.append(manis[1])
                rot_manifolds.append(manis[2])
                tor_manifolds.append(manis[3])
            else:
                out_manifolds.append(manis)

        if split_manifold:
            return (
                VMapManifolds(out_manifolds),
                VMapManifolds(f_manifolds),
                VMapManifolds(rot_manifolds),
                VMapManifolds(tor_manifolds),
                self.get_dims(dim_coords, max_num_atoms, max_num_rot_bonds),
            )
        else:
            return (
                VMapManifolds(out_manifolds),
                self.get_dims(dim_coords, max_num_atoms, max_num_rot_bonds),
            )


def test_product_space_manifold_getter():
    # Test the ProductSpaceManifoldGetter
    coord_manifold = "docking_flat_torus_01"
    rot_manifold = "sphere"
    tor_manifold = "docking_flat_torus_01"

    manifold_getter = ProductSpaceManifoldGetter(
        coord_manifold=coord_manifold,
        rot_manifold=rot_manifold,
        tor_manifold=tor_manifold,
    )

    batch_size = 2
    frac_coords = torch.randn(5 * batch_size, 3)
    rot = torch.randn(batch_size, 3)

    tor = torch.randn(7, 3)

    coord_batch = torch.repeat_interleave(
        torch.arange(batch_size),
        5,
    )
    tor_batch = torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    out = manifold_getter(
        frac_coords,
        rot,
        tor,
        coord_batch,
        tor_batch,
        split_manifold=True,
    )
    print(out)

    frac, rot, tor = manifold_getter.flatrep_to_georep(
        out.flat,
        out.dims,
        out.mask_f,
        out.mask_tor,
    )
    print(frac.shape, rot.shape, tor.shape)


def test_se3_manifold_getter():
    # Test the SE3ManifoldGetter
    coord_manifold = "docking_flat_torus_01"
    rot_manifold = "sphere"

    manifold_getter = SE3ManifoldGetter(
        coord_manifold=coord_manifold,
        rot_manifold=rot_manifold,
    )

    frac_coords = torch.randn((2, 3))
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    coord_batch = torch.tensor([0, 0])

    out = manifold_getter(
        frac_coords,
        rot,
        coord_batch,
        split_manifold=True,
    )
    print(out)

    frac, rot = manifold_getter.flatrep_to_georep(
        out.flat,
        out.dims,
        out.mask_f,
    )
    print(frac.shape, rot.shape)

    mani = manifold_getter.get_manifolds(
        torch.tensor([2]),
        3,
        split_manifold=True,
    )
    print(mani[2])


if __name__ == "__main__":
    test_se3_manifold_getter()
