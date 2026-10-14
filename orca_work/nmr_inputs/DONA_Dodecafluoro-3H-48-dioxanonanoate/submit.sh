#!/bin/bash
#SBATCH --job-name=DONA_Dodecafluor_nmr
#SBATCH --output=DONA_Dodecafluoro-3H-48-dioxanonanoate_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/DONA_Dodecafluoro-3H-48-dioxanonanoate/DONA_Dodecafluoro-3H-48-dioxanonanoate_nmr.inp > /home/Daniel.Barron/nmr_inputs/DONA_Dodecafluoro-3H-48-dioxanonanoate/DONA_Dodecafluoro-3H-48-dioxanonanoate_nmr.out
