# Run script for Pyflosic examples by LF.
# Type in the example file you want to run.

# Pyflosic path.
cd ..
pyflosic="$(dirname "$(pwd)")"
cd automatic_guessing
python3 $pyflosic/utils/init_guess_by_df.py Cu-II-planar.xyz -2 1 ccpvdz "PBE,PBE"
