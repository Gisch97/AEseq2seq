#!/bin/bash

# File paths
MODEL_FILE="src/seq2seq/model_unet_v2.py"
GLOBAL_CONFIG="config/global.json"
TRAIN_CONFIG="config/train.json"
TEST_CONFIG="config/test.json"

# Hyperparameters
NUM_CONV1=(1 2 3) 
NUM_CONV2=(1 2) 
# Base output path
BASE_OUTPUT_PATH="results/UNet_v2/convolution_layers"



echo "# Starting experiment at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:  NUM_CONV1 = ${NUM_CONV1[@]} NUM_CONV2 = ${NUM_CONV2[@]}"
echo "# Saving hyperparameters: NUM_CONV1 = ${NUM_CONV1[@]} NUM_CONV2 = ${NUM_CONV2[@]}" | tee -a "$BASE_OUTPUT_PATH/models.log"
echo "# Saving Model file content $MODEL_FILE >> $BASE_OUTPUT_PATH/models.log"
echo "# Model file content:" >> "$BASE_OUTPUT_PATH/models.log"
cat "$MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"

# Main script
for c1 in "${NUM_CONV1[@]}"; do
    for c2 in "${NUM_CONV2[@]}"; do
        # Construct save path and name
        save_name="UNet-num_convs-$c1-$c2"
        save_path="$BASE_OUTPUT_PATH/$save_name"

        echo "Executing: num_conv1=$c1; num_conv2=$c2"

        # Modify model configuration
        sed -i \
            -e "91s/num_conv1=[0-9e.-]*/num_conv1=$c1/" \
            -e "92s/num_conv2=[0-9e.-]*/num_conv2=$c2/" \
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