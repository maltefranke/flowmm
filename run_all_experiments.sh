#!/bin/bash

export data="dual_docking_data_distributed"
export model="docking_only_coords"

### docking
export vectorfield="dock_cspnet"

samplings=("uniform" "uniform_then_gaussian" "uniform_then_conformer" "voronoi_then_gaussian" "voronoi_then_conformer")

doOT=("true" "false")

for sampling in "${samplings[@]}"; do
    for ot in "${doOT[@]}"; do
        output_file="__Docking_${sampling}_ot_${ot}_%j.txt"
        export sampling=$sampling
        export ot=$ot
        sbatch --output="$output_file" run_docking.sh 
    done
done

### dual docking
export vectorfield="dual_docking_cspnet"

samplings=("uniform" "voronoi")
for sampling in "${samplings[@]}"; do
    for ot in "${doOT[@]}"; do
        output_file="__DualDocking_${sampling}_ot_${ot}_%j.txt"
        export sampling=$sampling
        export ot=$ot
        sbatch --output="$output_file" run_docking.sh
    done
done