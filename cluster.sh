#!/bin/bash
#SBATCH --job-name=yolov8
#SBATCH --ntasks=16
#SBATCH --mem=8192
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --tmp=100G
#SBATCH --mail-user=henry.marichal@fing.edu.uy

# de acuerdo a lo que quiera ejecutar puede elegir entre las siguientes tres líneas.
#SBATCH --gres=gpu:1 # se solicita una gpu cualquiera( va a tomar la primera que quede disponible indistintamente si es una p100 o una a100)


#SBATCH --partition=normal
#SBATCH --qos=gpu

source /etc/profile.d/modules.sh
source /clusteruy/home/henry.marichal/miniconda3/etc/profile.d/conda.sh
conda activate yolov8

# -------------------------------------------------------
#disco local SSD local al nodo. /clusteruy/home/henry.marichal se accede via NFS (puede ser realmente lento)
#el espacio local a utilizar se reserva con --tmp=XXXGb
LOCAL_NODE_DIR=/scratch/henry.marichal/

#other variables
RESULTADOS_DIR=$LOCAL_NODE_DIR/yolov8/resultados
DATASET_DIR=$LOCAL_NODE_DIR/yolov8
HOME_RESULTADOS_DIR=~/resultados/yolov8
HOME_DATASET_DIR=~/dataset_pith/yolo_urudendro
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

#copy dataset

check_command_result mkdir -p $DATASET_DIR

check_command_result mkdir -p $RESULTADOS_DIR

check_command_result cp  -r $HOME_DATASET_DIR $DATASET_DIR

#training model
cd ~/repos/yolov8/
./run_yolo.sh $RESULTADOS_DIR $DATASET_DIR/yolo_urudendro > "$stdout_file" 2> "$stderr_file"

#touch $RESULTADOS_DIR/resultado.txt


# -------------------------------------------------------
#copy results
mkdir -p $HOME_RESULTADOS_DIR
cp -r $RESULTADOS_DIR/* $HOME_RESULTADOS_DIR

#delete temporal files
rm -rf $RESULTADOS_DIR
rm -rf $DATASET_DIR