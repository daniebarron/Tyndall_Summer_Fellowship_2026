#!/bin/bash
#SBATCH --job-name=8-2_diPAP_nmr
#SBATCH --output=8-2_diPAP_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/8-2_diPAP/8-2_diPAP_nmr.inp > /home/Daniel.Barron/nmr_inputs/8-2_diPAP/8-2_diPAP_nmr.out
