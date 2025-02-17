#!/bin/bash
# Pasos para ejecutar un entrenamiento:
#   1) Modificar la ruta de MODEL_FILE
#   2) Modificar en __init__.py el nombre del modelo importado
#   3) Modificar en GLOBAL_JSON el nombre del experimento
#   4) Modifiar los BASE_OUTPUT_PATH y save_name a utilizar con la nomenclatura correcta. (check skip / no-skip)
#   5) Modifiar los parametros a utilizar y bucles
#   6) modificar lineas 26 y 27 (log de hiperparams) 

# File paths
MODEL_FILE="src/seq2seq/model_unet_v4.py"
GLOBAL_CONFIG="config/global.json"
TRAIN_CONFIG="config/train.json"
TEST_CONFIG="config/test.json"

# # Hyperparameters
NUM_CONV1=(2 3) 
NUM_CONV2=(2 3) 
FILTERS=(4 8)
RANK=(8 16)

LATENT_DIM=(8 16 32 64)
# Base output path
BASE_OUTPUT_PATH="results/UNet_v4/convolution_layers"
cp execute_Unet.sh "$BASE_OUTPUT_PATH"

### LOGGING execution
echo "# Starting experiment at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:  NUM_CONV1 = ${NUM_CONV1[@]} NUM_CONV2 = ${NUM_CONV2[@]} encode1_out = ${FILTERS[@]} encode2_out = ${RANK[@]} LATENT_DIM = ${LATENT_DIM[@]}"
echo "# Saving hyperparameters: NUM_CONV1 = ${NUM_CONV1[@]} NUM_CONV2 = ${NUM_CONV2[@]} encode1_out = ${FILTERS[@]} encode2_out = ${RANK[@]} LATENT_DIM = ${LATENT_DIM[@]}" | tee -a "$BASE_OUTPUT_PATH/models.log"
echo "# Saving Model file content $MODEL_FILE >> $BASE_OUTPUT_PATH/models.log"
echo "# Model file content:" >> "$BASE_OUTPUT_PATH/models.log"
cat "$MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"

# Main script
for c1 in "${NUM_CONV1[@]}"; do
    for c2 in "${NUM_CONV2[@]}"; do
        for f in "${FILTERS[@]}"; do
            for r in "${RANK[@]}"; do
                for ld in "${LATENT_DIM[@]}"; do
                    # Construct save path and name
                    save_name="UNet_v4_conv-num_convs-$c1-$c2-f$f-r$r-ld$ld"
                    save_path="$BASE_OUTPUT_PATH/$save_name"

                    echo "Executing: num_conv1=$c1; encode1_out=$f; encode2_out=$r; num_conv2=$c2; latent_dim=$ld"

                    # Modify model configuration
                    sed -i \
                        -e "91s/num_conv1=[0-9e.-]*/num_conv1=$c1/" \
                        -e "92s/num_conv2=[0-9e.-]*/num_conv2=$c2/" \
                        -e "98s/self.encode1_out=[0-9e.-]*/self.encode1_out=$f/" \
                        -e "100s/self.encode2_out=[0-9e.-]*/self.encode2_out=$r/" \
                        -e "109s/self.latent_dim=[0-9e.-]*/self.latent_dim=$ld/" \
                        "$MODEL_FILE"

                    # Update global configuration
                    echo "Updating global configuration... run: $save_name"
                    sed -i "s/\"run\": \"[^\"]*\"/\"run\": \"$save_name\"/" "$GLOBAL_CONFIG"

                    # Update train configuration
                    echo "Updating train configuration... out_path: $save_name"
                    sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"$save_path\"|" "$TRAIN_CONFIG"

                    # Train model
                    seq2seq train
                    echo "Training completed for configuration: $save_name"

                    # Update test configuration
                    echo "Updating test configuration... out_path: $save_name"
                    sed -i \
                        -e "s|\"model_weights\": \"[^\"]*\"|\"model_weights\": \"$save_path/weights.pmt\"|" \
                        -e "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"$save_path/test.csv\"|" \
                        "$TEST_CONFIG"

                    # Test model
                    seq2seq test
                    echo "Testing completed for configuration: $save_name"
        done
    done
done