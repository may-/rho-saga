#!/bin/bash
#


export SAGA=/home/students/ohta/saga        # project home
export DUMPDIR=$SAGA/weights                # where to save weights
export SAVEDIR=$SAGA/results                # where to save results
export DATADIR=$SAGA/data                   # input data root dir

# use python under $HOME/anaconda
export PATH=/home/students/ohta/anaconda2/bin:$PATH
#export PATH=/home/students/ohta/anaconda3/bin:$PATH

# main
/opt/slurm/bin/srun python $SAGA/multiclass.py -m $1 -s $2 -e $3 --learn_rate=$4 --data_name=$5 --seed=$6 --file_path=$DATADIR --weights_path=$DUMPDIR --results_path=$SAVEDIR


###################################################################################################################
# How to call this `train.sh` script on Slurm:
# - usage
#     $ /opt/slurm/bin/sbatch --output=<lof_file_name> --mem=<memoory_size> /path/to/train.sh <method_name> <solver_name> <num_epochs> <learning_rate> <data_name> <random_seed>
#
# - example
#     $ /opt/slurm/bin/sbatch --output=/home/students/ohta/saga/logs/rcv1/train.log --mem=500GB /home/students/ohta/saga/train.sh bandit saga 10 0.01 rcv1 $RANDOM
#
#
#
# Memory Size Hint for SAGA (for SGD, approx. half):
#     mnist           -> 16GB   [ca. 1.5 h for 100 epochs]
#     covtype         -> 128GB  [ca. h for 100 epochs]
#     news20          -> 256GB  [ca. 15 h for 100 epochs]
#     reuters4, rcv1  -> 500GB  [ca. h for 100 epochs]
###################################################################################################################
