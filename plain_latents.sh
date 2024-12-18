#!/bin/bash

# Array de valores para latent_dim
latent_dims=(12 16 20 32 64)

# Directorios y archivos de configuración
model_file="src/seq2seq/model.py"
global_config="config/global.json"
train_config="config/train.json"

# Bucle para cada valor de latent_dim
for dim in "${latent_dims[@]}"; do
  echo "Ejecutando con latent_dim=$dim"

  # Modificar el archivo model.py
  sed -i "s/latent_dim=[0-9]*/latent_dim=$dim/" "$model_file"

  # Modificar el archivo global.json
  sed -i "s/\"run\": \"latent[0-9]*-valid02\"/\"run\": \"latent$dim-valid02\"/" "$global_config"
  
  # Modificar el archivo train.json
  sed -i "s/latent-[0-9]*/latent-$dim/" "$train_config"

  # Ejecutar el entrenamiento
  seq2seq train

  echo "Finalizado latent_dim=$dim"
done
