#!/bin/bash
#SBATCH --job-name=DPOSA_Capstone_A_nmr
#SBATCH --output=DPOSA_Capstone_A_6-2_FTNO_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/DPOSA_Capstone_A_6-2_FTNO/DPOSA_Capstone_A_6-2_FTNO_nmr.inp > /home/Daniel.Barron/nmr_inputs/DPOSA_Capstone_A_6-2_FTNO/DPOSA_Capstone_A_6-2_FTNO_nmr.out
