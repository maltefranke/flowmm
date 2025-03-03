import os
from ase.io import read, write
from ase import Atoms, Atom
from ase.visualize import view
from pymatgen.core.lattice import Lattice
import torch
import pandas as pd
import numpy as np
import argparse
import math
import tempfile
from rdkit import Chem
from rdkit.Chem import Draw

from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer
from ovito.modifiers import CreateBondsModifier

from diffcsp.common.data_utils import lattice_params_to_matrix_torch
from flowmm.rfm.manifolds.flat_torus import FlatTorus01
from rfm_docking.reassignment import ot_reassignment
from rfm_docking.featurization import get_atoms_and_pos
from diffcsp.common.data_utils import (
    lattice_params_to_matrix,
)


class InMemoryTrajectory:
    def __init__(self):
        self.frames = []

    def write(self, atoms):
        self.frames.append(atoms)

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)


def create_gif_from_traj(
    traj_file,
    output_gif,
    structure_type="osda_pred_and_osda_target",
):
    data = torch.load(traj_file, map_location="cpu")
    lattice = data["lattices"]  # gen_trajectory
    lattice = lattice_params_to_matrix_torch(
        lattice[:3].view(1, -1), lattice[3:].view(1, -1)
    )
    lattice = lattice.squeeze()

    loading = data["loading"]

    osda = data["osda"]
    osda_atoms = osda["atom_types"]
    osda_target_coords = osda["frac_coords"]
    osda_conformers = osda["conformer_cart"]

    zeolite = data["zeolite"]
    zeolite_atoms = zeolite["atom_types"]
    zeolite_coords = zeolite["frac_coords"]

    atoms_per_mol = len(osda_conformers) // loading
    atoms_per_mol = torch.repeat_interleave(atoms_per_mol, loading)

    mol_ids = torch.repeat_interleave(
        torch.arange(loading.sum(), device=atoms_per_mol.device),
        atoms_per_mol,
    )

    structures = []
    for frame_idx, (f, rot) in enumerate(zip(data["f_s"], data["rot_s"])):
        # rotate the osda conformers
        conformer_rot_t_cart = torch.einsum("nij,nj->ni", rot[mol_ids], osda_conformers)

        # translate the osda conformers
        inv_lattice_mat = torch.inverse(torch.tensor(lattice, dtype=torch.float32))

        conformer_rot_t_frac = torch.einsum(
            "ni,ij->ni", conformer_rot_t_cart, inv_lattice_mat
        )

        frac_coords_i = conformer_rot_t_frac + f[mol_ids]
        frac_coords_i = frac_coords_i % 1.0

        if structure_type == "osda_pred_and_none_target":
            atoms_i = osda_atoms
            coords_i = frac_coords_i
        elif structure_type == "none_pred_and_osda_target":
            atoms_i = osda_atoms
            coords_i = osda_target_coords
        elif structure_type == "osda_pred_and_osda_target":
            atoms_i = torch.cat([osda_atoms, osda_atoms])
            coords_i = torch.cat([frac_coords_i, osda_target_coords])
        elif structure_type == "all_pred_and_none_target":
            atoms_i = torch.cat([osda_atoms, zeolite_atoms])
            coords_i = torch.cat([frac_coords_i, zeolite_coords])
        elif structure_type == "all_target":
            atoms_i = torch.cat([osda_atoms, zeolite_atoms])
            coords_i = torch.cat([osda_target_coords, zeolite_coords])

        predicted = Atoms(
            atoms_i, scaled_positions=coords_i, cell=tuple(lattice.squeeze().tolist())
        )
        # save as CIF file
        # write(output_gif.split(".")[0] + f"_{structure_type}_{frame_idx}.cif", predicted, format="cif")

        structures.append(predicted)

    write(
        output_gif.split(".")[0] + f"_{structure_type}_final.cif",
        structures[-1],
        format="cif",
    )
    traj = InMemoryTrajectory()
    for atoms in structures:
        traj.write(atoms)

    write(
        output_gif.split(".")[0] + ".gif",
        traj,
        rotation="0x,0y,0z",
        interval=200,
    )

    write(
        output_gif.split(".")[0] + "_303030.gif",
        traj,
        rotation="30x,30y,30z",
        interval=200,
    )
    write(
        output_gif.split(".")[0] + "_606060.gif",
        traj,
        rotation="60x,60y,60z",
        interval=200,
    )
    print(f"GIF saved as {output_gif}")


if __name__ == "__main__":
    traj_file = "runs/trash/2025-03-02/21-02-45/se3_docking-se3_dock_cspnet-7ck6b0d8/156559915_traj.pt"

    create_gif_from_traj(
        traj_file,
        "runs/trash/2025-03-02/21-02-45/se3_docking-se3_dock_cspnet-7ck6b0d8/138480949_traj.pt",
        structure_type="all_pred_and_none_target",
    )
