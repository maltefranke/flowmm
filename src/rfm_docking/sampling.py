import torch
from torch_geometric.data import Batch
import math
import numpy as np
from diffcsp.common.data_utils import (
    cart_to_frac_coords,
)
from rfm_docking.utils import smiles_to_pos


def get_sigma(
    sigma_in_A: torch.Tensor,
    lattice_lenghts: torch.Tensor,
    num_atoms: torch.Tensor,
) -> torch.Tensor:
    """Transform sigma in angstroms to sigma in fractional space for each atom."""
    sigma_in_A = (
        sigma_in_A
        * (1 / math.sqrt(3))  # makes sure vector length is sigma_in_A
        * torch.ones((num_atoms.sum(), 3), device=lattice_lenghts.device)
    )
    batch = torch.repeat_interleave(
        torch.arange(num_atoms.shape[0], device=lattice_lenghts.device), num_atoms
    )

    # sigma in fractional space
    sigma = sigma_in_A / lattice_lenghts[batch]

    return sigma


def sample_mol_in_frac(
    smiles,
    lattice_lengths,
    lattice_angles,
    loading,
    forcefield="mmff",
    regularized=True,
    device="cpu",
):
    """Sample a molecule in a fractional space."""

    all_pos = []
    num_atoms = []
    for smi, load in zip(smiles, loading):
        # Convert SMILES to 3D coordinates
        pos = smiles_to_pos(smi, forcefield, device)

        all_pos.append(pos.repeat(load, 1))
        num_atoms.append(pos.shape[0] * load)

    all_pos = torch.cat(all_pos, dim=0).reshape(-1, 3).to(device)
    num_atoms = torch.tensor(num_atoms).to(device)

    # Convert cartesian coordinates to fractional coordinates
    frac = cart_to_frac_coords(
        all_pos,
        lattice_lengths,
        lattice_angles,
        num_atoms=num_atoms,
        regularized=regularized,
    )

    return frac


def sample_uniform(osda: Batch, loading: torch.Tensor) -> torch.Tensor:
    """Sample from a uniform distribution and expand to N atoms per molecule."""
    num_mols = loading.sum()
    atoms_per_mol = osda.num_atoms // loading

    # assignment for each molecule
    mol_ids = torch.arange(num_mols, device=osda.frac_coords.device)
    atom_ids = torch.repeat_interleave(atoms_per_mol, loading)

    ids = torch.repeat_interleave(mol_ids, atom_ids)

    uniform = torch.rand((num_mols, 3))

    prior = uniform[ids]

    return prior


def sample_uniform_then_gaussian(
    osda: Batch, loading: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """Sample from a uniform distribution and then apply a Gaussian noise."""
    uniform = sample_uniform(osda, loading)
    gaussian = torch.randn_like(osda.frac_coords)

    prior = gaussian * sigma + uniform

    return prior


def sample_uniform_then_conformer(
    osda: Batch, smiles: list[str], loading: torch.Tensor
) -> torch.Tensor:
    """Sample from a uniform distribution and then set the molecular conformer."""
    uniform = sample_uniform(osda, loading)

    conformers = osda.conformer

    prior = conformers + uniform

    return prior


def sample_voronoi(
    num_atoms: torch.Tensor,
    batch_indices: torch.Tensor,
    voronoi_nodes: torch.Tensor,
    num_voronoi_nodes: torch.Tensor,
    loading: torch.Tensor,
) -> torch.Tensor:
    """Sample from a Voronoi diagram."""

    atoms_per_mol = num_atoms // loading
    voronoi_nodes_batch = torch.repeat_interleave(
        torch.arange(num_voronoi_nodes.shape[0]), num_voronoi_nodes
    )

    prior = []
    for i in batch_indices.unique():
        loading_i = loading[i]
        voronoi_nodes_i = voronoi_nodes[voronoi_nodes_batch == i]

        # draw random voronoi nodes based on loading
        random_indices = torch.randperm(voronoi_nodes_i.shape[0])
        draw_voronoi_nodes = voronoi_nodes_i[random_indices, :][:loading_i, :]

        if draw_voronoi_nodes.shape[0] / (loading_i) < 1:
            # in case there are fewer voronoi nodes than molecules, should be extremely rare
            difference = loading_i - draw_voronoi_nodes.shape[0]
            draw_voronoi_nodes = torch.cat(
                [draw_voronoi_nodes, draw_voronoi_nodes[:difference]], dim=0
            )

        # expand to match the number of atoms in the molecule
        draw_voronoi_nodes = torch.repeat_interleave(
            draw_voronoi_nodes, atoms_per_mol[i], dim=0
        )

        prior.append(draw_voronoi_nodes)

    prior = torch.cat(prior, dim=0)

    return prior
