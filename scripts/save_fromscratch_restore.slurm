#!/bin/bash
#SBATCH -A ylab
#SBATCH -J pancirc-save
#SBATCH -p gpu
#SBATCH -D /media/share/workdir/1610305236/Panfungi/4_model
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -o /home/1610305236/log/%x-%j.out
#SBATCH -e /home/1610305236/log/%x-%j.err

bash scripts/save_fromscratch_restore.sh
