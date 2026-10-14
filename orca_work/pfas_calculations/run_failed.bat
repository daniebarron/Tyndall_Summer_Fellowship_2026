@echo off
echo Rerunning 9 failed frequency calculations...
echo.

echo [1/9] Running Freq for: PFPrS_Perfluoropropanesulfonic_acid
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\PFPrS_Perfluoropropanesulfonic_acid"
"C:\Orca_6.1.1\orca.exe" "PFPrS_Perfluoropropanesulfonic_acid_freq.inp" > "PFPrS_Perfluoropropanesulfonic_acid_freq.out"
echo Finished: PFPrS_Perfluoropropanesulfonic_acid
echo.

echo [2/9] Running Freq for: PFEESA_Perfluoro-2-ethoxyethane-sulfonic_acid
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\PFEESA_Perfluoro-2-ethoxyethane-sulfonic_acid"
"C:\Orca_6.1.1\orca.exe" "PFEESA_Perfluoro-2-ethoxyethane-sulfonic_acid_freq.inp" > "PFEESA_Perfluoro-2-ethoxyethane-sulfonic_acid_freq.out"
echo Finished: PFEESA_Perfluoro-2-ethoxyethane-sulfonic_acid
echo.

echo [3/9] Running Freq for: MeFOSE_N-methylperfluorooctanesulfonamido-ethanol
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\MeFOSE_N-methylperfluorooctanesulfonamido-ethanol"
"C:\Orca_6.1.1\orca.exe" "MeFOSE_N-methylperfluorooctanesulfonamido-ethanol_freq.inp" > "MeFOSE_N-methylperfluorooctanesulfonamido-ethanol_freq.out"
echo Finished: MeFOSE_N-methylperfluorooctanesulfonamido-ethanol
echo.

echo [4/9] Running Freq for: MeFOSAA_N-methylperfluorooctanesulfonamid-HAc
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\MeFOSAA_N-methylperfluorooctanesulfonamid-HAc"
"C:\Orca_6.1.1\orca.exe" "MeFOSAA_N-methylperfluorooctanesulfonamid-HAc_freq.inp" > "MeFOSAA_N-methylperfluorooctanesulfonamid-HAc_freq.out"
echo Finished: MeFOSAA_N-methylperfluorooctanesulfonamid-HAc
echo.

echo [5/9] Running Freq for: 8-2_FTUCA_2H-Perfluoro-2-decenoic_acid
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\8-2_FTUCA_2H-Perfluoro-2-decenoic_acid"
"C:\Orca_6.1.1\orca.exe" "8-2_FTUCA_2H-Perfluoro-2-decenoic_acid_freq.inp" > "8-2_FTUCA_2H-Perfluoro-2-decenoic_acid_freq.out"
echo Finished: 8-2_FTUCA_2H-Perfluoro-2-decenoic_acid
echo.

echo [6/9] Running Freq for: 6-2_FTAB_Capstone_B
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\6-2_FTAB_Capstone_B"
"C:\Orca_6.1.1\orca.exe" "6-2_FTAB_Capstone_B_freq.inp" > "6-2_FTAB_Capstone_B_freq.out"
echo Finished: 6-2_FTAB_Capstone_B
echo.

echo [7/9] Running Freq for: 6-2_FTS_Fluorotelomer_sulfonate
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\6-2_FTS_Fluorotelomer_sulfonate"
"C:\Orca_6.1.1\orca.exe" "6-2_FTS_Fluorotelomer_sulfonate_freq.inp" > "6-2_FTS_Fluorotelomer_sulfonate_freq.out"
echo Finished: 6-2_FTS_Fluorotelomer_sulfonate
echo.

echo [8/9] Running Freq for: 6-2_diPAP
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\6-2_diPAP"
"C:\Orca_6.1.1\orca.exe" "6-2_diPAP_freq.inp" > "6-2_diPAP_freq.out"
echo Finished: 6-2_diPAP
echo.

echo [9/9] Running Freq for: 3-3_FTCA_Fluorotelomer_carboxylic_acid
cd /d "C:\Users\danie\Downloads\orca_work\pfas_calculations\3-3_FTCA_Fluorotelomer_carboxylic_acid"
"C:\Orca_6.1.1\orca.exe" "3-3_FTCA_Fluorotelomer_carboxylic_acid_freq.inp" > "3-3_FTCA_Fluorotelomer_carboxylic_acid_freq.out"
echo Finished: 3-3_FTCA_Fluorotelomer_carboxylic_acid
echo.

echo All done!
pause