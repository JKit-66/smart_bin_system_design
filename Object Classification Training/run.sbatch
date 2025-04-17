#!/bin/bash
####################################
#  Iridis 5 slurm script template  
#                                  
#  Submit script: sbatch filename  
#                                  
####################################
#SBATCH --ntasks=40     # Number of processor cores (i.e. tasks)
#SBATCH --nodes=1     # Number of nodes requested
#SBATCH --ntasks-per-node=40     # Tasks per node
#SBATCH --cpus-per-task=1     # Threads per task
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jkl1a20@soton.ac.uk
#SBATCH --time=60:00:00   # walltime

echo "Job started at: $(date)"

EXECUTABLE=$mainfs/scratch/jkl1a20/newfolder/roboflow/trainYOLO.py
WORKDIR=/mainfs/scratch/jkl1a20/newfolder/results

module load conda
module load ultralytics
module load matplotlib
#matplotlib.use('TkAgg')

cd $WORKDIR
python3 $EXECUTABLE $WORKDIR &> out8_20_3.txt


echo "Job ended at: $(date)"