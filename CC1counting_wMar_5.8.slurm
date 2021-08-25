#!/bin/sh
#SBATCH --partition=general-compute --qos=general-compute --gres=gpu:2
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --job-name="CC1counting_wMar_5.8"
#SBATCH --output=srun_CC1counting_wMar_5.8_%A_%a.out
#SBATCH --mail-user=cooper36@buffalo.edu
#SBATCH --mail-type=ALL

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}

cd $SLURM_SUBMIT_DIR
echo "working directory = "$SLURM_SUBMIT_DIR

ulimit -s unlimited
#

tstart=`date`
echo "###### start time: $tstart"

tic=`date +%s`

#
module load tensorflow/1.12.0-gpu-py36
python3 -u CC1counting_wMar_5.8.py > results_CC1counting_wMar_5.8.txt
		
toc=`date +%s`
tend=`date`
echo "###### end time: $tend"

elapsedTime=`expr $toc - $tic`
echo "Elapsed Time = $elapsedTime seconds"

#
echo "All Done!"
