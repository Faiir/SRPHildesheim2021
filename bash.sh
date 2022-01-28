#!/usr/bin/env bash
#SBATCH --job-name=test_robust_active_learning_1
#SBATCH --output=test_robust_active_learning_1%j.log
#SBATCH --error=test_robust_active_learning_1%j.err
#SBATCH --mail-user=kraussn@uni-hildesheim.de
#SBATCH --partition=TEST
#SBATCH --gres=gpu:0
set -e
source /home/kraussn/anaconda3/bin/activate /home/kraussn/anaconda3/envs/robustal

cd /home/kraussn/robust_active_learning/robust_active_learning  # navigate to the directory if necessary

srun /home/kraussn/anaconda3/envs/robustal/bin/python3 -m start_experiment -c experiment_settings.json