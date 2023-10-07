#!/bin/bash
#SBATCH --job-name=yolov8
#SBATCH --ntasks=16
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --tmp=100G
#SBATCH --mail-user=henry.marichal@fing.edu.uy

# de acuerdo a lo que quiera ejecutar puede elegir entre las siguientes tres líneas.
#SBATCH --gres=gpu:1 # se solicita una gpu cualquiera( va a tomar la primera que quede disponible indistintamente si es una p100 o una a100)


#SBATCH --partition=normal
#SBATCH --qos=gpu

source /etc/profile.d/modules.sh
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate yolov8

# -------------------------------------------------------
#disco local SSD local al nodo. /clusteruy/home/henry.marichal se accede via NFS (puede ser realmente lento)
#el espacio local a utilizar se reserva con --tmp=XXXGb
LOCAL_NODE_DIR=/scratch/henry.marichal/

#other variables
DATASET=forest_zoom_in
NODE_RESULTADOS_DIR=$LOCAL_NODE_DIR/yolov8/resultados
NODE_DATASET_DIR=$LOCAL_NODE_DIR/yolov8
HOME_RESULTADOS_DIR=~/resultados/yolov8
HOME_DATASET_DIR=~/dataset_pith/TreeTrace_Douglas_format/$DATASET
stdout_file="$HOME_RESULTADOS_DIR/stdout.txt"
stderr_file="$HOME_RESULTADOS_DIR/stderr.txt"
# Define a function to check the result of a command
check_command_result() {
    # Run the command passed as an argument
    "$@"

    # Check the exit status
    if [ $? -eq 0 ]; then
        echo "Command was successful."
    else
        echo "Command failed with an error."
        exit 1
    fi
}

####Prepare directories
rm -rf $NODE_DATASET_DIR
rm -rf $NODE_RESULTADOS_DIR
rm -rf $HOME_RESULTADOS_DIR

check_command_result mkdir -p $NODE_DATASET_DIR
check_command_result mkdir -p $NODE_RESULTADOS_DIR
check_command_result mkdir -p $HOME_RESULTADOS_DIR

####Move dataset to node local disk
check_command_result cp  -r $HOME_DATASET_DIR $NODE_DATASET_DIR


# -------------------------------------------------------
# Run the program

cd ~/repos/yolov8/
./run_yolo.sh $NODE_RESULTADOS_DIR $NODE_DATASET_DIR/$DATASET > "$stdout_file" 2> "$stderr_file"


# -------------------------------------------------------
#copy results to HOME
mkdir -p $HOME_RESULTADOS_DIR
cp -r $NODE_RESULTADOS_DIR/* $HOME_RESULTADOS_DIR
cp -r $NODE_DATASET_DIR/* $HOME_RESULTADOS_DIR
#delete temporal files
rm -rf $NODE_RESULTADOS_DIR
rm -rf $NODE_DATASET_DIR