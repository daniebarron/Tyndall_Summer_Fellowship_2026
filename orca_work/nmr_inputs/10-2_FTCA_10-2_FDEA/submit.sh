#!/bin/bash
#SBATCH --job-name=10-2_FTCA_10-2_F_nmr
#SBATCH --output=10-2_FTCA_10-2_FDEA_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/10-2_FTCA_10-2_FDEA/10-2_FTCA_10-2_FDEA_nmr.inp > /home/Daniel.Barron/nmr_inputs/10-2_FTCA_10-2_FDEA/10-2_FTCA_10-2_FDEA_nmr.out
