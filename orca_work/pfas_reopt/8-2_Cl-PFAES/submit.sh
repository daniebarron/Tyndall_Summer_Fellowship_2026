#!/bin/bash
#SBATCH --job-name=ClPFAES_reopt
#SBATCH --output=ClPFAES_reopt.log
#SBATCH --ntasks=4
#SBATCH --mem=14G
#SBATCH --time=24:00:00
#SBATCH --partition=cpuq-tyndall

module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
INPDIR=/home/Daniel.Barron/pfas_reopt/8-2_Cl-PFAES

$ORCA $INPDIR/8-2_Cl-PFAES_11-chloroeicosafluoro-3-oxaundecane-1-sulfonate.inp > $INPDIR/8-2_Cl-PFAES_11-chloroeicosafluoro-3-oxaundecane-1-sulfonate.out
