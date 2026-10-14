#!/bin/bash
#SBATCH --job-name=iso_8_2_FTOH_ir
#SBATCH --output=/home/Daniel.Barron/pfas_new/iso_8_2_FTOH/iso_8_2_FTOH_ir.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --partition=cpuq-tyndall

module load openmpi/gcc/64/4.1.5

ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
INPDIR=/home/Daniel.Barron/pfas_new/iso_8_2_FTOH

export PATH=/home/Daniel.Barron/software/orca_j5sY:$PATH
export LD_LIBRARY_PATH=/home/Daniel.Barron/software/orca_j5sY:$LD_LIBRARY_PATH

$ORCA $INPDIR/iso_8_2_FTOH_ir.inp > $INPDIR/iso_8_2_FTOH_ir.out
