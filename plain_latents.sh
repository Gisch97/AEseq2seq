#!/bin/bash

# Directorios y archivos de configuración
model_file="src/seq2seq/model.py"
global_config="config/global.json"
train_config="config/train.json"

# Configuración de hiperparámetros
learning_rates=(1e-4 1e-3 1e-2)
output_thresholds=(0.5 0.7)

# Bucle para cada combinación de hiperparámetros
for lr in "${learning_rates[@]}"; do
  for out_th in "${output_thresholds[@]}"; do
    echo "Ejecutando con lr=$lr y output_th=$out_th"

    # Modificar el archivo model.py
    sed -i "39s/lr=[0-9.e-]*/lr=$lr/" "$model_file"
    sed -i "41s/output_th=[0-9.]*,/output_th=$out_th,/" "$model_file"

    # Modificar el archivo global.json
    sed -i "s/\"run\": \"[^\"]*\"/\"run\": \"latent-32-lr$lr-ot$out_th\"/" "$global_config"

    # Modificar el archivo train.json
    sed -i "s|\"out_path\": \"[^\"]*\"|\"out_path\": \"results/plain_latent32/lr$lr-ot$out_th\"|" "$train_config"

    # Ejecutar el entrenamiento
    seq2seq train
    echo "Finalizado con lr=$lr y output_th=$out_th"
  done
done
