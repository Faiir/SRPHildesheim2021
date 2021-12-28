#!/usr/bin/env bash
#SBATCH --job-name=robust_active_learning_1
#SBATCH --output=robust_active_learning_1%j.log
#SBATCH --error=robust_active_learning_1%j.err
#SBATCH --mail-user=kraussn@uni-hildesheim.de
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

cd ~/robust_active_learning          # navigate to the directory if necessary
source activate pytorchenv
srun python -m experiment_setup.py -c experiment_setup.json        # python jobs require the srun command to work

