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
echo "Cloning the repo"
cd $TMPDIR/
rm -rf $TMPDIR/deeplearning_demos
git clone https://github.com/jeremyfix/deeplearning_demos.git
cd deeplearning_demos

echo ""
echo "Creating the virtualenv"

python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate
ls
python -m pip install -r requirements.txt
python -m pip install .

echo ""
echo "Running"

dldemos_server --config config://detectron2-panopticseg.yaml

if [[ $? != 0 ]]; then
    exit -1
fi

