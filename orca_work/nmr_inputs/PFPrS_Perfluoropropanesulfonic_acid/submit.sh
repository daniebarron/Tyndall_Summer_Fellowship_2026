#!/bin/bash
#SBATCH --job-name=PFPrS_Perfluorop_nmr
#SBATCH --output=PFPrS_Perfluoropropanesulfonic_acid_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/PFPrS_Perfluoropropanesulfonic_acid/PFPrS_Perfluoropropanesulfonic_acid_nmr.inp > /home/Daniel.Barron/nmr_inputs/PFPrS_Perfluoropropanesulfonic_acid/PFPrS_Perfluoropropanesulfonic_acid_nmr.out
