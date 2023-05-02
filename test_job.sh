#!/bin/bash
#SBATCH -J testing-nccl
#SBATCH -C gpu
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --time 10
#SBATCH --qos debug

module load pytorch

GPUSPERNODE=4
export MASTER_ADDR=`hostname`
export MASTER_PORT=12345

for i in `seq 1 100`
do
    echo "trial $i..."
    srun -u -l --cpu-bind=none ./run_test.sh
done
