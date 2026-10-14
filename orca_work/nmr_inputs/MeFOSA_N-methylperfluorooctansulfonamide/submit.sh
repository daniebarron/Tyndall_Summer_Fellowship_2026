#!/bin/bash
#SBATCH --job-name=MeFOSA_N-methylp_nmr
#SBATCH --output=MeFOSA_N-methylperfluorooctansulfonamide_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/MeFOSA_N-methylperfluorooctansulfonamide/MeFOSA_N-methylperfluorooctansulfonamide_nmr.inp > /home/Daniel.Barron/nmr_inputs/MeFOSA_N-methylperfluorooctansulfonamide/MeFOSA_N-methylperfluorooctansulfonamide_nmr.out
