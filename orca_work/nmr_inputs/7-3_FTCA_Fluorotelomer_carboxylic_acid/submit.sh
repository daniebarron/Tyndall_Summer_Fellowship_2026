#!/bin/bash
#SBATCH --job-name=7-3_FTCA_Fluorot_nmr
#SBATCH --output=7-3_FTCA_Fluorotelomer_carboxylic_acid_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/7-3_FTCA_Fluorotelomer_carboxylic_acid/7-3_FTCA_Fluorotelomer_carboxylic_acid_nmr.inp > /home/Daniel.Barron/nmr_inputs/7-3_FTCA_Fluorotelomer_carboxylic_acid/7-3_FTCA_Fluorotelomer_carboxylic_acid_nmr.out
