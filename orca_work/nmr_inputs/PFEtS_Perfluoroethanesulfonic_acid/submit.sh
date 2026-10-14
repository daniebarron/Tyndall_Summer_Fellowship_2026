#!/bin/bash
#SBATCH --job-name=PFEtS_Perfluoroe_nmr
#SBATCH --output=PFEtS_Perfluoroethanesulfonic_acid_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/PFEtS_Perfluoroethanesulfonic_acid/PFEtS_Perfluoroethanesulfonic_acid_nmr.inp > /home/Daniel.Barron/nmr_inputs/PFEtS_Perfluoroethanesulfonic_acid/PFEtS_Perfluoroethanesulfonic_acid_nmr.out
