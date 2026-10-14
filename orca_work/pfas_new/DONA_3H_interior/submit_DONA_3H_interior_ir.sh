#!/bin/bash
#SBATCH --job-name=DONA_3H_interior_ir
#SBATCH --output=/home/Daniel.Barron/pfas_new/DONA_3H_interior/DONA_3H_interior_ir.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --partition=cpuq-tyndall

module load openmpi/gcc/64/4.1.8

ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
INPDIR=/home/Daniel.Barron/pfas_new/DONA_3H_interior

export PATH=/home/Daniel.Barron/software/orca_j5sY:$PATH
export LD_LIBRARY_PATH=/home/Daniel.Barron/software/orca_j5sY:$LD_LIBRARY_PATH

$ORCA $INPDIR/DONA_3H_interior_ir.inp > $INPDIR/DONA_3H_interior_ir.out
