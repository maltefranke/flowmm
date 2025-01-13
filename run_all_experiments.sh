#!/bin/bash

export data="dual_docking_data_distributed"

### docking
export model="dock_cspnet"
export vectorfield="dock_cspnet"

samplings=("uniform" "uniform_then_gaussian" "uniform_then_conformer" "voronoi_then_gaussian" "voronoi_then_conformer")

doOT=("true" "false")

for sampling in "${samplings[@]}"; do
    for ot in "${doOT[@]}"; do
        export sampling=$sampling
        export ot=$ot
        sbatch run_docking.sh 
    done
done

### dual docking
export model="dual_docking_cspnet"
export vectorfield="dual_docking_cspnet"

samplings=("uniform" "voronoi")
for sampling in "${samplings[@]}"; do
    for ot in "${doOT[@]}"; do
        export sampling=$sampling
        export ot=$ot
        sbatch run_docking.sh
    done
done