#!/bin/bash
#SBATCH --job-name=diPAP_reopt
#SBATCH --output=diPAP_reopt.log
#SBATCH --ntasks=2
#SBATCH --mem=14G
#SBATCH --time=72:00:00
#SBATCH --partition=cpuq-tyndall

module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
INPDIR=/home/Daniel.Barron/pfas_reopt/8-2_diPAP

$ORCA $INPDIR/8-2_diPAP_reopt.inp > $INPDIR/8-2_diPAP_reopt.out
