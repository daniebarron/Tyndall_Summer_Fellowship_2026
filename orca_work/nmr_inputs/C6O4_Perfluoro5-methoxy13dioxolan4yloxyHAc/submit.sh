#!/bin/bash
#SBATCH --job-name=C6O4_Perfluoro5-_nmr
#SBATCH --output=C6O4_Perfluoro5-methoxy13dioxolan4yloxyHAc_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/C6O4_Perfluoro5-methoxy13dioxolan4yloxyHAc/C6O4_Perfluoro5-methoxy13dioxolan4yloxyHAc_nmr.inp > /home/Daniel.Barron/nmr_inputs/C6O4_Perfluoro5-methoxy13dioxolan4yloxyHAc/C6O4_Perfluoro5-methoxy13dioxolan4yloxyHAc_nmr.out
