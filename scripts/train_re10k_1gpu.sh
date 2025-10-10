#!/bin/bash

#SBATCH --job-name=training_key
#SBATCH --output=/scratch/u5aa/jeffreyhu.u5aa/Code/diffusion-forcing-transformer/sbatch_output/%j_%x.out
#SBATCH --error=/scratch/u5aa/jeffreyhu.u5aa/Code/diffusion-forcing-transformer/sbatch_output/%j_%x.err
#SBATCH --nodes=2                # num nodes
#SBATCH --ntasks-per-node=2      # 4 tasks per node (one per GPU)
#SBATCH --gpus-per-node=2        # 4 GPUs per node
#SBATCH --time=00:05:00         # Hours:Mins:Secs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jhh57@cam.ac.uk

hostname
nvidia-smi --list-gpus

module load cuda/12.6
source $HOME/miniforge3/bin/activate dfot

cd $SCRATCH/Code/diffusion-forcing-transformer

python -m main \
    +name=RE10k \
    dataset=realestate10k \
    algorithm=dfot_video_pose \
    experiment=video_generation \
    @diffusion/continuous
