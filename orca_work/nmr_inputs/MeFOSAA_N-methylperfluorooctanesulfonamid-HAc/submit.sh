#!/bin/bash
#SBATCH --job-name=MeFOSAA_N-methyl_nmr
#SBATCH --output=MeFOSAA_N-methylperfluorooctanesulfonamid-HAc_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/MeFOSAA_N-methylperfluorooctanesulfonamid-HAc/MeFOSAA_N-methylperfluorooctanesulfonamid-HAc_nmr.inp > /home/Daniel.Barron/nmr_inputs/MeFOSAA_N-methylperfluorooctanesulfonamid-HAc/MeFOSAA_N-methylperfluorooctanesulfonamid-HAc_nmr.out
