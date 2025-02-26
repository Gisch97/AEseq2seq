#!/bin/bash
# Pasos para ejecutar un entrenamiento:
#   1) Modificar la ruta de MODEL_FILE
#   2) Modificar en __init__.py el nombre del modelo importado
#   3) Modificar en GLOBAL_JSON el nombre del experimento
#   4) Modifiar los BASE_OUTPUT_PATH y save_name a utilizar con la nomenclatura correcta. (check skip / no-skip)
#   5) Modifiar los parametros a utilizar y bucles
#   6) modificar lineas 26 y 27 (log de hiperparams) 


## NUM_CONV1 = 1,2
## num_layers = 0,1,2
## NUM_CONV2 = 0,1,2,3
## FLAT = 0, 1
## skip = 0, 1

### EXPERIMENTO Y MODELOS SELECCIONADOS
# EXPERIMENT_NAME="test"
EXPERIMENT_NAME="UNet_selection_v4p_e1"
MODEL_NAME="unet_v4_p_skip_e1"
MODEL_NAME_NO_SKIP="unet_v4_p_no_skip_e1"

# File paths
MODEL_PATH="src/seq2seq/models/unet_selection"

INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"
MODEL_FILE_NO_SKIP="$MODEL_PATH/$MODEL_NAME_NO_SKIP.py"

BASE_OUTPUT_PATH="results/UNet_selection/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"
# # Hyperparameters
MAX_EPOCHS=20
NUM_CONV1=(1 2) 
NUM_CONV2=(1 2 3) 
RESNET_LAYERS=(0 1 2)  

### LOGGING execution
cp execute_progressive_selection.sh "$BASE_OUTPUT_PATH"
echo "# Starting experiment -- $EXPERIMENT_NAME -- at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:   NUM_CONV1 = ${NUM_CONV1[@]} NUM_CONV2 = ${NUM_CONV2[@]} RESNET_LAYERS = ${RESNET_LAYERS[@]} |2 ENCODERS BLOCKS SKIP CONNS (0 1)"
echo "#Saving hyperparameters:  NUM_CONV1 = ${NUM_CONV1[@]} NUM_CONV2 = ${NUM_CONV2[@]} RESNET_LAYERS = ${RESNET_LAYERS[@]} |2 ENCODERS BLOCKS SKIP CONNS (0 1)" | tee -a "$BASE_OUTPUT_PATH/models.log"
echo "# Saving Model file content $MODEL_FILE >> $BASE_OUTPUT_PATH/models.log"
echo "# Model file content:" >> "$BASE_OUTPUT_PATH/models.log"
cat "$MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"

# from .models.unet_selection.unet_v4_p_skip import seq2seq
# Main script
sed -i "s/\"exp\": \"[^\"]*\"/\"exp\": \"$EXPERIMENT_NAME\"/" "$GLOBAL_CONFIG"
sed -i "s/\"max_epochs\": [0-9e.-]*/\"max_epochs\": $MAX_EPOCHS/" "$GLOBAL_CONFIG" 

for c1 in "${NUM_CONV1[@]}"; do
    
    for c2 in "${NUM_CONV2[@]}"; do

        for resnet in "${RESNET_LAYERS[@]}"; do 
            base_name="A_nc$c1-resnet$resnet-nc$c2"
            ###############################################################################
            ### SKIP CONNECTIONS 
            ###############################################################################
            echo "----------------  SKIP CONNECTIONS----------------"
            echo "MODIFY INIT $INIT , ADD MODEL $MODEL_FILE"
            sed -i "16s/.models.unet_selection.*/.models.unet_selection.$MODEL_NAME import seq2seq/" "$INIT"
       
            echo "Executing: num_conv1=$c1; num_conv2=$c2; resnet=$resnet skip=1"
            # Modify model configuration
            sed -i \
                -e "87s/num_layers=[0-9e.-]*/num_layers=$resnet/" \
                -e "93s/num_conv1=[0-9e.-]*/num_conv1=$c1/" \
                -e "94s/num_conv2=[0-9e.-]*/num_conv2=$c2/" \
                "$MODEL_FILE"

            save_name="$base_name-2encode-flat1-skip1"
            echo "save_name: $save_name"
            bash train_test.sh "$BASE_OUTPUT_PATH" "$save_name"
            ###############################################################################
            ### NO SKIP CONNECTIONS 
            ###############################################################################
            echo "----------------NO SKIP CONNECTIONS----------------"
            echo "MODIFY INIT $INIT ADD MODEL $MODEL_FILE_NO_SKIP"
            sed -i "16s/.models.unet_selection.*/.models.unet_selection.$MODEL_NAME_NO_SKIP import seq2seq/" "$INIT"

            echo "Executing: num_conv1=$c1; num_conv2=$c2; resnet=$resnet skip=0"
            # Modify model configuration
            sed -i \
                -e "87s/num_layers=[0-9e.-]*/num_layers=$resnet/" \
                -e "93s/num_conv1=[0-9e.-]*/num_conv1=$c1/" \
                -e "94s/num_conv2=[0-9e.-]*/num_conv2=$c2/" \
                "$MODEL_FILE_NO_SKIP"

            save_name="$base_name-2encode-flat1-skip0" 
            echo "save_name: $save_name"
            bash train_test.sh "$BASE_OUTPUT_PATH" "$save_name"

        done
    done 

done 