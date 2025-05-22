#!/bin/bash
# ====== User-configurable section ======
CPUS=1                            # Number of CPUs (per task)
GPUS=0                            # Number of GPUs
MEM=4G                            # Memory per node
TIME="02:00:00"                   # Time limit (hh:mm:ss) / maximum time to run is 72 hours
PARTITION=cpu                    # Partition to use (e.g., cpu, gpu_h100)
PYTHON_SCRIPT="src/test.py"           # Python file to execute (must be in current dir or use full path)
ARGS=""

DIR="/home/ds/ds_ds/ds_wi22230/LLM_GAN_MarketResearch/LLM_GAN_MarketResearch"
VENV_PATH="$DIR/llm-gan"          # Path to your virtualenv
LOGDIR="$DIR/logs"
mkdir -p "$LOGDIR"

JOBNAME="synthetic_data_job_$(date +%Y%m%d%H%M%S)"
LOGFILE="$LOGDIR/$JOBNAME.log"

# ====== SLURM job submission ======
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --output=$LOGFILE
#SBATCH --error=$LOGFILE
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME

# Go to script directory (optional, adjust if needed)
cd $DIR

# Activate virtual environment
source $VENV_PATH/bin/activate

# Run Python script
echo "Starting Python script at \$(date)"
python $PYTHON_SCRIPT $ARGS
echo "Finished at \$(date)"
EOF

echo -e "\nJob '$JOBNAME' submitted to partition '$PARTITION'. Logs: $LOGFILE"
