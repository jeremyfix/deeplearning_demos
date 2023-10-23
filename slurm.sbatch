#!/bin/bash 

#SBATCH --job-name=deeplearning_demos
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00 

current_dir=`pwd`

# Fix env variables as bashrc and profile are not loaded
export LOCAL=$HOME/.local
export PATH=$PATH:$LOCAL/bin

echo "Running on $(hostname)"

echo ""
echo "Virtual environment"

python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

echo ""
echo "Installation of dlserver"
python -m pip install git+https://github.com/jeremyfix/deeplearning_demos.git#subdirectory=dlserver

echo ""
echo "Running"

dlserver

if [[ $? != 0 ]]; then
    exit -1
fi

