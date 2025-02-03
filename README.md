# MatDock: Multi-molecule docking in porous materials with flow matching

# Installation

Follow the steps to get the files:

```bash
# clone this repo into the folder flowmm/
cd flowmm
git submodule init
git submodule update
bash create_env_file.sh  # creates the necessary .env file
```

The submodules include [CDVAE](https://arxiv.org/abs/2110.06197), [DiffCSP](https://arxiv.org/abs/2309.04475), and [Riemannian Flow Matching](https://arxiv.org/abs/2302.03660).

Now we can install `flowmm`. We recommend using `micromamba` because conda is extremely slow. You can install `micromamba` by [following their guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html#automatic-install). If the installation fails, try again a few more times.

```bash
micromamba env create -f environment.yml
```

Activate using

```bash
micromamba activate flowmm
```


# Data

The training data is in `.csv` format in `data/`. When you first train on it, the script will convert the data into a faster-to-load format.


# Experiments


There are a few preselected `model` options in `scripts_model/conf/model`.
Since we assume known atom types and fixed lattices, we only have to distinguish coordinate manifolds.
We have
- `docking_only_coords` - translation equivariant target VF. Use for task where some positions are fixed (e.g. crystal atoms), and some are predicted (e.g. molecule to dock)
- `optimize_only_coords` - translation invariant target VF. Use for task where **all** positions are predicted

For the `vectorfield` options in `scripts_model/conf/vectorfield`, we have the following choices:
- `dock_cspnet` - all-atom docking
- `dual_docking_cspnet` - dock center of mass first, then dock corresponding atoms
- `optimize_cspnet` - all-atom structure optimization


## Evaluation

Discussion about evaluation is limited to FlowMM. `scripts_model/evaluate.py` uses `click`, allowing it to serve as a multi-purpose evaluation program.

# Citation

If you find this repository helpful for your publications, please consider citing our paper:

```
@inproceedings{
    miller2024flowmm,
    title={Flow{MM}: Generating Materials with Riemannian Flow Matching},
    author={Benjamin Kurt Miller and Ricky T. Q. Chen and Anuroop Sriram and Brandon M Wood},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=W4pB7VbzZI}
}
```

# License

`flowmm` is CC-BY-NC licensed, as found in the `LICENSE.md` file. However, the git submodules may have different license terms:
- `cdvae`: MIT License
- `DiffCSP-official`: MIT License
- `riemmanian-fm`: [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/)

The licenses for the dependencies can be viewed at the corresponding project's homepage.
