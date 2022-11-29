#!/bin/sh
#SBATCH --job-name="behavioral_clustering"
#SBATCH --account="ie-imf"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00-20:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=80000
#SBATCH --mail-user="ulrik_danielsen@hotmail.com"
#SBATCH --mail-type=ALL

WORDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE"
echo "Total of $SLURM_NTASKS cores"


module purge
module load Python/3.10.4-GCCcore-11.3.0

RUNPATH=/cluster/work/ulrikbd/specialization_project/code/
cd $RUNPATH
source bc/bin/activate
bc/bin/python3.10 idun.py
