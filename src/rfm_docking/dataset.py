import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData
from pymatgen.core.lattice import Lattice
from omegaconf import ValueNode

from tqdm import tqdm
from multiprocessing import Pool
from diffcsp.common.data_utils import (
    lattice_params_to_matrix,
)
from rfm_docking.featurization import (
    get_atoms_and_pos,
    split_zeolite_and_osda_pos,
    featurize_osda,
    get_feature_dims,
)
from rfm_docking.utils import gen_edges, get_osda_mean_pbc, smiles_to_pos
from rfm_docking.voronoi_utils import get_voronoi_nodes, cluster_voronoi_nodes


def process_one_(args):
    row, prop_list, zeolite_params = args

    crystal_id = row.dock_crystal

    ### process the lattice
    lattice_matrix = eval(row.dock_lattice)
    lattice_matrix = np.array(lattice_matrix)

    lattice_lengths = Lattice(lattice_matrix).lengths
    lattice_lengths = torch.tensor(lattice_lengths)
    lattice_angles = Lattice(lattice_matrix).angles
    lattice_angles = torch.tensor(lattice_angles)

    # lattice has to conform to pymatgen's Lattice object, rotate data accordingly
    lattice_matrix_target = lattice_params_to_matrix(
        *Lattice(lattice_matrix).parameters
    )

    M = lattice_matrix.T @ lattice_matrix_target
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # Ensure R is a proper rotation matrix with determinant 1
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1  # Correct for reflection if necessary
        R = U @ Vt

    R = torch.tensor(R, dtype=torch.float32)

    inv_lattice = torch.inverse(
        torch.tensor(lattice_matrix_target, dtype=torch.float32)
    )

    smiles = row.smiles

    conformer = smiles_to_pos(smiles, forcefield="mmff", device="cpu")
    # conformer to fractional coordinates
    conformer_frac = conformer @ inv_lattice.T

    loading = int(row.loading)
    node_feats, edge_feats, edge_index = featurize_osda(smiles)

    osda_node_feats = node_feats.repeat(loading, 1)

    _, N = edge_index.shape

    # Create offsets using torch.arange
    offsets = torch.arange(loading, device=edge_index.device) * (
        osda_node_feats.shape[0] // loading
    )
    # Repeat the offsets for each edge entry
    offsets = offsets.view(-1, 1).repeat(1, N).view(1, -1)  # Shape: (1, N * loading)

    # increment edge indices for each loaded osda
    osda_edge_indices = edge_index.repeat(1, loading)
    osda_edge_indices += offsets

    osda_edge_feats = edge_feats.repeat(loading, 1)

    ### process the docked structures
    dock_axyz = eval(row.dock_xyz)

    # zeolite and osda are separated
    dock_zeolite_axyz, dock_osda_axyz = split_zeolite_and_osda_pos(
        dock_axyz, smiles, loading
    )
    # process the zeolite
    dock_zeolite_atoms, dock_zeolite_pos = get_atoms_and_pos(dock_zeolite_axyz)
    # change to correct coordinate system
    dock_zeolite_pos = dock_zeolite_pos @ R

    # cartesian to fractional, wrap back into unit cell
    dock_zeolite_pos = (dock_zeolite_pos @ inv_lattice) % 1.0

    # calculate the voronoi nodes
    # remove oxygen atoms before for smoother voronoi nodes
    non_oxygen = np.where(dock_zeolite_atoms.squeeze() != 8, True, False)
    non_o_zeolite_pos = dock_zeolite_pos[non_oxygen]

    voronoi_nodes = get_voronoi_nodes(
        non_o_zeolite_pos, lattice_matrix_target, cutoff=2.5
    )
    voronoi_nodes = cluster_voronoi_nodes(
        voronoi_nodes, lattice_matrix_target, cutoff=13.0, merge_tol=1.0
    )

    dock_zeolite_edges, _ = gen_edges(
        num_atoms=torch.tensor(dock_zeolite_atoms.shape[0]).view(1, 1),
        frac_coords=dock_zeolite_pos,
        lattices=torch.tensor(Lattice(lattice_matrix).parameters).view(1, -1),
        node2graph=torch.zeros(dock_zeolite_atoms.shape[0], dtype=torch.long),
        edge_style=zeolite_params["edge_style"],
        radius=zeolite_params["cutoff"],
        max_neighbors=zeolite_params["max_neighbors"],
        self_edges=zeolite_params["self_edges"],
    )

    dock_zeolite_graph_arrays = (
        dock_zeolite_pos,
        dock_zeolite_atoms,
        lattice_lengths,
        lattice_angles,
        voronoi_nodes,
        dock_zeolite_edges,
        dock_zeolite_atoms.shape[0],
    )

    # process the osda
    dock_osda_atoms, dock_osda_pos = get_atoms_and_pos(dock_osda_axyz)

    # remove hydrogens
    non_hydrogen = torch.where(dock_osda_atoms.squeeze() != 1, True, False)
    dock_osda_atoms = dock_osda_atoms[non_hydrogen]
    dock_osda_pos = dock_osda_pos[non_hydrogen]

    dock_osda_pos = dock_osda_pos @ R

    # cartesian to fractional, wrap back into unit cell
    dock_osda_pos = (dock_osda_pos @ inv_lattice) % 1.0

    # center of mass (com) of the osda taking into account periodic boundary conditions
    dock_osda_com_frac_pbc = get_osda_mean_pbc(
        dock_osda_pos, lattice_matrix_target, osda_edge_indices, loading
    )

    dock_osda_graph_arrays = (
        dock_osda_pos,
        dock_osda_atoms,
        lattice_lengths,
        lattice_angles,
        dock_osda_com_frac_pbc,
        osda_edge_indices,
        dock_osda_atoms.shape[0],
    )

    ### process the optimized structures
    opt_axyz = eval(row.opt_xyz)
    opt_zeolite_axyz, opt_ligand_axyz = split_zeolite_and_osda_pos(
        opt_axyz, smiles, loading
    )
    opt_zeolite_atoms, opt_zeolite_pos = get_atoms_and_pos(opt_zeolite_axyz)
    opt_zeolite_pos = opt_zeolite_pos @ R
    opt_zeolite_pos = (opt_zeolite_pos @ inv_lattice) % 1.0

    opt_zeolite_graph_arrays = (
        opt_zeolite_pos,
        opt_zeolite_atoms,
        lattice_lengths,
        lattice_angles,
        voronoi_nodes,
        #  NOTE below we assume that opt_zeolite is close to dock_zeolite --> has same edges, save computation
        dock_zeolite_edges,
        opt_zeolite_atoms.shape[0],
    )

    opt_osda_atoms, opt_osda_pos = get_atoms_and_pos(opt_ligand_axyz)

    # remove hydrogens
    non_hydrogen = torch.where(opt_osda_atoms != 1)[0]
    opt_osda_atoms = opt_osda_atoms[non_hydrogen]
    opt_osda_pos = opt_osda_pos[non_hydrogen]

    opt_osda_pos = opt_osda_pos @ R
    opt_osda_pos = (opt_osda_pos @ inv_lattice) % 1.0

    opt_osda_graph_arrays = (
        opt_osda_pos,
        opt_osda_atoms,
        lattice_lengths,
        lattice_angles,
        osda_edge_indices,
        opt_osda_atoms.shape[0],
    )

    properties = dict()
    for k in prop_list:
        if k in row.keys():
            properties[k] = torch.tensor(row[k], dtype=torch.float64)

    preprocessed_dict = {
        "crystal_id": crystal_id,
        "smiles": smiles,
        "conformer": conformer_frac,
        "loading": loading,
        "osda_feats": (osda_node_feats, osda_edge_feats, osda_edge_indices),
        "dock_zeolite_graph_arrays": dock_zeolite_graph_arrays,
        "dock_osda_graph_arrays": dock_osda_graph_arrays,
        "opt_zeolite_graph_arrays": opt_zeolite_graph_arrays,
        "opt_osda_graph_arrays": opt_osda_graph_arrays,
    }
    preprocessed_dict.update(properties)
    return preprocessed_dict


def process_one(args):
    try:
        return process_one_(args)
    except Exception as e:
        print(f"Error processing crystal {args[0].dock_crystal}")
        print(e)
        return None


def custom_preprocess(
    input_file,
    num_workers,
    prop_list,
    zeolite_params,
):
    df = pd.read_csv(input_file)  # .loc[:0]

    def parallelized():
        # Create a pool of workers
        with Pool(num_workers) as pool:
            for item in tqdm(
                pool.imap_unordered(
                    process_one,
                    iterable=[
                        (df.iloc[idx], prop_list, zeolite_params)
                        for idx in range(len(df))
                    ],
                    chunksize=20,
                ),
                total=len(df),
            ):
                yield item

    # NOTE uncomment to debug process_one
    # process_one((df.iloc[0], graph_method, prop_list))

    # Convert the unordered results to a list
    unordered_results = list(parallelized())

    # remove Nones from the list
    unordered_results = [result for result in unordered_results if result is not None]

    """# Create a dictionary mapping crystal_id to results
    mpid_to_results = {result["crystal_id"]: result for result in unordered_results}

    # Create a list of ordered results based on the original order of the dataframe
    ordered_results = [
        mpid_to_results[df.iloc[idx]["dock_crystal"]] for idx in range(len(df))
    ]

    return ordered_results"""

    # we're fine with unordered results
    return unordered_results


def custom_add_scaled_lattice_prop(
    data_list, lattice_scale_method, graph_arrays_key="dock_zeolite_graph_arrays"
):
    for dict in data_list:
        graph_arrays = dict[graph_arrays_key]
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == "scale_length":
            lengths = lengths / float(num_atoms) ** (1 / 3)

        dict["scaled_lattice"] = np.concatenate([lengths, angles])


def dict_to_data(data_dict, task, prop_list, scaler, node_feat_dims):
    # scaler is set in DataModule set stage
    prop = dict()
    for p in prop_list:
        # print(p, type(data_dict[p]))
        prop[p] = scaler[p].transform(data_dict[p]).view(1, -1).to(dtype=torch.float32)
    (
        frac_coords,
        atom_types,
        lengths,
        angles,
        com_frac_pbc,
        _,  # NOTE edge indices will be overwritten with rdkit featurization
        num_atoms,
    ) = data_dict["dock_osda_graph_arrays"]

    smiles = data_dict["smiles"]
    loading = data_dict["loading"]
    conformer = data_dict["conformer"]

    osda_node_feats, osda_edge_feats, osda_edge_indices = data_dict["osda_feats"]

    # atom_coords are fractional coordinates
    # edge_index is incremented during batching
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    osda_data = Data(
        frac_coords=torch.Tensor(frac_coords),
        atom_types=torch.LongTensor(atom_types),
        lengths=torch.Tensor(lengths).view(1, -1),
        angles=torch.Tensor(angles).view(1, -1),
        edge_index=torch.LongTensor(
            osda_edge_indices
        ).contiguous(),  # shape (2, num_edges)
        edge_feats=osda_edge_feats,
        node_feats=osda_node_feats,
        center_of_mass=com_frac_pbc,
        num_atoms=num_atoms,
        num_bonds=osda_edge_indices.shape[0],
        num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        # y=prop.view(1, -1), # TODO mrx prop is now a dict so this will fail
    )

    (
        frac_coords,
        atom_types,
        lengths,
        angles,
        voronoi_nodes,
        edge_indices,
        num_atoms,
    ) = data_dict["dock_zeolite_graph_arrays"]

    # assign the misc class to the zeolite node feats, except for atom types
    zeolite_node_feats = torch.tensor(node_feat_dims) - 1
    zeolite_node_feats = zeolite_node_feats.repeat(num_atoms, 1)
    zeolite_node_feats[:, 0] = atom_types

    zeolite_data = Data(
        frac_coords=torch.Tensor(frac_coords),
        atom_types=torch.LongTensor(atom_types),
        lengths=torch.Tensor(lengths).view(1, -1),
        angles=torch.Tensor(angles).view(1, -1),
        edge_index=torch.LongTensor(edge_indices).contiguous(),  # shape (2, num_edges)
        voronoi_nodes=voronoi_nodes,
        node_feats=zeolite_node_feats,
        num_atoms=num_atoms,
        num_bonds=edge_indices.shape[0],
        num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        num_voronoi_nodes=voronoi_nodes.shape[0],
    )

    if "optimize" in task:
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,  # NOTE edge indices will be overwritten with rdkit featurization
            num_atoms,
        ) = data_dict["opt_osda_graph_arrays"]

        osda_data_opt = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                osda_edge_indices
            ).contiguous(),  # shape (2, num_edges)
            edge_feats=osda_edge_feats,
            node_feats=osda_node_feats,
            num_atoms=num_atoms,
            num_bonds=osda_edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )

        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            voronoi_nodes,
            edge_indices,
            num_atoms,
        ) = data_dict["opt_zeolite_graph_arrays"]

        zeolite_data_opt = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices
            ).contiguous(),  # shape (2, num_edges)
            voronoi_nodes=voronoi_nodes,
            node_feats=zeolite_node_feats,
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            num_voronoi_nodes=voronoi_nodes.shape[0],
        )

    data = HeteroData()
    data.crystal_id = data_dict["crystal_id"]
    data.smiles = smiles
    data.loading = loading
    data.conformer = conformer
    data.osda = osda_data
    data.zeolite = zeolite_data
    if "optimize" in task:
        data.osda_opt = osda_data_opt
        data.zeolite_opt = zeolite_data_opt
    data.num_atoms = osda_data.num_atoms + zeolite_data.num_atoms
    data.lengths = data.zeolite.lengths
    data.angles = data.zeolite.angles
    data.y = prop  # NOTE dictionary

    return data


class CustomCrystDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        path: ValueNode,
        prop: ValueNode,
        preprocess_workers: ValueNode,
        lattice_scale_method: ValueNode,
        save_path: ValueNode,
        task: ValueNode,
        zeolite_edges: ValueNode,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.lattice_scale_method = lattice_scale_method
        self.task = task

        node_feat_dims, edge_feat_dims = get_feature_dims()
        self.node_feat_dims = node_feat_dims
        self.edge_feat_dims = edge_feat_dims

        self.zeolite_edges = zeolite_edges

        self.preprocess(
            save_path, preprocess_workers, prop, self.zeolite_edges
        )  # prop = ['bindingatoms']

        # add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        # TODO not sure if how to scale osda lattice - should it be different from zeolite, or scale their combined cell?
        custom_add_scaled_lattice_prop(
            self.cached_data, lattice_scale_method, "dock_zeolite_graph_arrays"
        )
        custom_add_scaled_lattice_prop(
            self.cached_data, lattice_scale_method, "dock_osda_graph_arrays"
        )

        if "optimize" in task:
            custom_add_scaled_lattice_prop(
                self.cached_data, lattice_scale_method, "opt_zeolite_graph_arrays"
            )
            custom_add_scaled_lattice_prop(
                self.cached_data, lattice_scale_method, "opt_osda_graph_arrays"
            )

        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop, zeolite_params):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = custom_preprocess(
                self.path,
                preprocess_workers,
                prop_list=prop,
                zeolite_params=zeolite_params,
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        data = dict_to_data(
            data_dict, self.task, self.prop, self.scaler, self.node_feat_dims
        )
        return data

    def __repr__(self) -> str:
        return f"CustomCrystDataset({self.name=}, {self.path=})"


class DistributedDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        raw_data_dir: ValueNode,
        processed_data_dir: ValueNode,
        preprocess_workers: ValueNode,
        prop: ValueNode,
        lattice_scale_method: ValueNode,
        task: ValueNode,
        zeolite_edges: ValueNode,
    ):
        """
        Initialize the dataset.
        Args:
            file_paths (list of str): Paths to raw data files.
            cache_dir (str): Directory to store processed files.
        """
        self.name = name
        self.task = task
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)

        node_feat_dims, edge_feat_dims = get_feature_dims()
        self.node_feat_dims = node_feat_dims
        self.edge_feat_dims = edge_feat_dims

        self.prop = prop

        file_paths = [
            os.path.join(self.raw_data_dir, file)
            for file in os.listdir(self.raw_data_dir)
        ]

        # Process and store data
        self.processed_files = []
        self.num_data_points = []
        for file_path in file_paths:
            processed_file = os.path.join(
                processed_data_dir,
                f"{os.path.splitext(os.path.basename(file_path))[0]}.pt",
            )
            if not os.path.exists(processed_file):
                print(f"Processing {file_path}...")
                cached_data = custom_preprocess(
                    file_path,
                    preprocess_workers,
                    prop_list=prop,
                    zeolite_params=zeolite_edges,
                )

                custom_add_scaled_lattice_prop(
                    cached_data, lattice_scale_method, "dock_zeolite_graph_arrays"
                )
                custom_add_scaled_lattice_prop(
                    cached_data, lattice_scale_method, "dock_osda_graph_arrays"
                )

                if "optimize" in task:
                    custom_add_scaled_lattice_prop(
                        cached_data,
                        lattice_scale_method,
                        "opt_zeolite_graph_arrays",
                    )
                    custom_add_scaled_lattice_prop(
                        cached_data, lattice_scale_method, "opt_osda_graph_arrays"
                    )

                os.makedirs(os.path.dirname(processed_file), exist_ok=True)
                torch.save(cached_data, processed_file)

                self.num_data_points.append(len(cached_data))

            else:
                # Get the number of data points in the processed file
                cached_data = torch.load(processed_file)
                self.num_data_points.append(len(cached_data))
                # remove cached data from memory
                del cached_data

            self.processed_files.append(processed_file)

        # Create a cumulative index map for efficient global indexing
        self.cumulative_sizes = np.cumsum(self.num_data_points)

        self.scaler = None
        self.lattice_scaler = None

    def __len__(self):
        return sum(self.num_data_points)

    def __getitem__(self, idx):
        # Find the file that contains the requested index
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
        if file_idx > 0:
            idx_in_file = idx - self.cumulative_sizes[file_idx - 1]
        else:
            idx_in_file = idx

        # Load data from the file
        data_dict_list = torch.load(self.processed_files[file_idx])
        data_dict = data_dict_list[idx_in_file]
        del data_dict_list

        data = dict_to_data(
            data_dict, self.task, self.prop, self.scaler, self.node_feat_dims
        )

        return data

    def __repr__(self) -> str:
        return f"DistributedDataset({self.name=}, {self.path=})"


if __name__ == "__main__":
    path = "data/test_data.csv"

    data = pd.read_csv(path)
    data = data.dropna(
        subset=[
            "dock_crystal",
            "dock_lattice",
            "smiles",
            "dock_xyz",
            "opt_xyz",
            "loading",
        ]
    )

    crystal_id = data.dock_crystal.tolist()

    lattices = data.dock_lattice.apply(eval).tolist()
    smiles = data.smiles.tolist()

    dock_xyz = data.dock_xyz.apply(eval).tolist()
    opt_xyz = data.opt_xyz.apply(eval).tolist()

    cols = data.columns
