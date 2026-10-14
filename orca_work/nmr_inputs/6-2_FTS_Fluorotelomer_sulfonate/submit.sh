#!/bin/bash
#SBATCH --job-name=6-2_FTS_Fluorote_nmr
#SBATCH --output=6-2_FTS_Fluorotelomer_sulfonate_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/6-2_FTS_Fluorotelomer_sulfonate/6-2_FTS_Fluorotelomer_sulfonate_nmr.inp > /home/Daniel.Barron/nmr_inputs/6-2_FTS_Fluorotelomer_sulfonate/6-2_FTS_Fluorotelomer_sulfonate_nmr.out
