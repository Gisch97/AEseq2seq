#!/bin/bash

# File paths
MODEL_FILE="src/seq2seq/model_unet.py"
GLOBAL_CONFIG="config/global.json"
TRAIN_CONFIG="config/train.json"
TEST_CONFIG="config/test.json"

# Hyperparameters
NUM_CONV1=(1 2 3)
STRIDE1=(1 2)
NUM_CONV2=(1 2 3)
STRIDE2=(1 2)

# Base output path
BASE_OUTPUT_PATH="results/UNet/add_convolution_layers"

# Main script
for c1 in "${NUM_CONV1[@]}"; do
    for c2 in "${NUM_CONV2[@]}"; do
        for s1 in "${STRIDE1[@]}"; do 
            for s2 in "${STRIDE2[@]}"; do
                # Construct save path and name
                save_name="UNet-num_convs-$c1-$c2-stride-$s1-$s2"
                save_path="$BASE_OUTPUT_PATH/$save_name"

                echo "Executing: num_conv1=$c1, stride1=$s1; num_conv2=$c2, stride2=$s2"

                # Modify model configuration
                sed -i \
                    -e "89s/stride_1=[0-9e.-]*/stride_1=$s1/" \
                    -e "90s/stride_2=[0-9e.-]*/stride_2=$s2/" \
                    -e "91s/num_conv1=[0-9e.-]*/num_conv1=$c1/" \
                    -e "92s/num_conv2=[0-9e.-]*/num_conv2=$c2/" \
                    "$MODEL_FILE"

                # Update global configuration
                sed -i "s/\"run\": \"[^\"]*\"/\"run\": \"$save_name\"/" "$GLOBAL_CONFIG"

                # Update train configuration
                sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"$save_path\"|" "$TRAIN_CONFIG"

                # Train model
                seq2seq train
                echo "Training completed for configuration: $save_name"

                # Update test configuration
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
done