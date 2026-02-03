#!/bin/bash 

#SBATCH --job-name=deeplearning_demos
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00 

current_dir=`pwd`

# Fix env variables as bashrc and profile are not loaded
export LOCAL=$HOME/.local
export PATH=$PATH:$LOCAL/bin

echo "Running on $(hostname)"

echo ""
echo "Virtual environment"

uv venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

echo "Installation of the dependencies"

uv pip install PyYAML opencv-python numpy matplotlib transformers 
uv pip install coloredlogs flatbuffers numpy packaging protobuf sympy
uv pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

echo ""
echo "Installation of dlserver"
# python -m pip install git+https://github.com/jeremyfix/deeplearning_demos.git#subdirectory=dlserver
uv pip install ./dlserver

echo ""
echo "Running"

dlserver

if [[ $? != 0 ]]; then
    exit -1
fi

