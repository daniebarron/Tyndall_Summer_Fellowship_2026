#!/bin/bash
#SBATCH --job-name=6_2_FTAB_protonated_nmr
#SBATCH --output=/home/Daniel.Barron/pfas_new/6_2_FTAB_protonated/6_2_FTAB_protonated_nmr.log
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall

module load openmpi/gcc/64/4.1.8

ORCA=/home/Daniel.Barron/software/orca_j5sY/orca

export PATH=/home/Daniel.Barron/software/orca_j5sY:$PATH
export LD_LIBRARY_PATH=/home/Daniel.Barron/software/orca_j5sY:$LD_LIBRARY_PATH

$ORCA /home/Daniel.Barron/pfas_new/6_2_FTAB_protonated/6_2_FTAB_protonated_nmr.inp > /home/Daniel.Barron/pfas_new/6_2_FTAB_protonated/6_2_FTAB_protonated_nmr.out
