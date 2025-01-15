#!/bin/bash

# Directorios y archivos de configuración
model_file="src/seq2seq/model_unet.py"
global_config="config/global.json"
train_config="config/train.json"
test_config="config/test.json"

run="UNet-k3-filters-4-8-200e"
# Configuración de hiperparámetros
stride1=(2)
stride2=(2 1) 

# Bucle para cada combinación de hiperparámetros
for s1 in "${stride1[@]}"; do 
    for s2 in "${stride2[@]}"; do
        echo "... Ejecutando con stride1 =$s1 y stride2=$s2"
        echo "... Guardando en $run-stride-$s1$s2"
        # Modificar el archivo model.py 
        sed -i "89s/stride_1=[0-9e.-]*/stride_1=$s1/" "$model_file"
        sed -i "90s/stride_2=[0-9e.-]*/stride_2=$s2/" "$model_file"

        # Modificar el archivo global.json
        sed -i "s/\"run\": \"[^\"]*\"/\"run\": \"$run-stride-$s1$s2\"/" "$global_config"

        # Modificar el archivo train.json
        sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"results/$run-stride-$s1$s2\"|" "$train_config"


        # Ejecutar el entrenamiento
        seq2seq train
        echo "... Entrenamiento Finalizado con stride1 =$s1 y stride2=$s2"

        # Modificar el archivo test.json
        sed -i "s|\"model_weights\": \"[^\"]*\"|\"model_weights\": \"results/$run-stride-$s1$s2/weights.pmt\"|" "$test_config"
        sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"results/$run-stride-$s1$s2/test.csv\"|" "$test_config"
        

        # Ejecutar el entrenamiento
        seq2seq test
        echo "Prueba Finalizada (Unet kernel = 3) con s1=$s1, s2=$s2"
    done
done