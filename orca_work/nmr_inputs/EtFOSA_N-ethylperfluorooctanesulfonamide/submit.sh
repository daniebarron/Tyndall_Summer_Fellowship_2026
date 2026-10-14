#!/bin/bash
#SBATCH --job-name=EtFOSA_N-ethylpe_nmr
#SBATCH --output=EtFOSA_N-ethylperfluorooctanesulfonamide_nmr.log
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpuq-tyndall
module load openmpi/gcc/64/4.1.5
ORCA=/home/Daniel.Barron/software/orca_j5sY/orca
$ORCA /home/Daniel.Barron/nmr_inputs/EtFOSA_N-ethylperfluorooctanesulfonamide/EtFOSA_N-ethylperfluorooctanesulfonamide_nmr.inp > /home/Daniel.Barron/nmr_inputs/EtFOSA_N-ethylperfluorooctanesulfonamide/EtFOSA_N-ethylperfluorooctanesulfonamide_nmr.out
