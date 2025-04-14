#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
EXPERIMENT_NAME="Unet_paper_noise_0_1"
MODEL_NAME="unet_original" 

# File paths
MODEL_PATH="src/seq2seq/models/paper_based"
INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"

BASE_OUTPUT_PATH="results/UNet_selection/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"

# Hyperparameters
MAX_EPOCHS=20
pool_mode=('max')
up_mode=('upsample')
addition=('cat')
skip=(0 1)

p=(0.0 0.03125 0.0625 0.09375 0.125 0.15625 0.1875 0.21875 0.25 0.28125 0.3125 0.34375 0.375 0.40625 0.4375 0.46875 0.5 0.53125 0.5625 0.59375 0.625 0.65625 0.6875 0.71875 0.75 0.78125 0.8125 0.84375 0.875 0.90625 0.9375 0.96875 1.0)

### LOGGING de la ejecución
cp fold_it.sh "$BASE_OUTPUT_PATH"
echo "# Starting experiment -- $EXPERIMENT_NAME -- at $(date)" > "$BASE_OUTPUT_PATH/models.log"
echo "Saving hyperparameters:" \
     "pool mode = ${pool_mode[@]}," \
     "up mode = ${up_mode[@]}," \
     "addition = ${addition[@]}," \
     "skip = ${skip[@]}" | tee -a "$BASE_OUTPUT_PATH/models.log"
 
echo "# Config train #" >> "$BASE_OUTPUT_PATH/models.log"
cat config/train.json >> "$BASE_OUTPUT_PATH/models.log"
echo "# Config test #" >> "$BASE_OUTPUT_PATH/models.log"
cat config/test.json >> "$BASE_OUTPUT_PATH/models.log"

echo "# Guardando contenido del archivo del modelo: $MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"
echo "# Contenido del modelo:" >> "$BASE_OUTPUT_PATH/models.log"
cat "$MODEL_FILE" >> "$BASE_OUTPUT_PATH/models.log"

# Actualiza el archivo de configuración global
sed -i "s/\"exp\": \"[^\"]*\"/\"exp\": \"$EXPERIMENT_NAME\"/" "$GLOBAL_CONFIG"
sed -i "s/\"max_epochs\": [0-9e.-]*/\"max_epochs\": $MAX_EPOCHS/" "$GLOBAL_CONFIG"
 
# Se configura el valor de skip fijo en 0 para este bloque (se puede anidar otro bucle si se desea variar)

echo "Ejecutando: paper_skip=0"
base_name="paper_"
save_name="${base_name}skip0"
# # Modificar la configuración del modelo para pool, up y skip
sed -i \
    -e "84s/\(skip=\)[0-9e.-]*/skip=0/" \
    "$MODEL_FILE"

bash scripts/train.sh "$BASE_OUTPUT_PATH" "$save_name" 
for swaps in ${p[@]}; do
    bash scripts/test_w_swap.sh "$BASE_OUTPUT_PATH" "$save_name" "$swaps"
done

echo "Ejecutando: paper_skip=1"
# Actualizar el parámetro addition en el modelo
sed -i \
    -e "84s/\(skip=\)[0-9e.-]*/skip=1/" \
    "$MODEL_FILE"
# Definir el nombre de guardado usando base_name y el valor de addition
save_name="${base_name}skip1"
echo "save_name: $save_name"

bash scripts/train.sh "$BASE_OUTPUT_PATH" "$save_name" 
for swaps in ${p[@]}; do
    bash scripts/test_w_swap.sh "$BASE_OUTPUT_PATH" "$save_name" "$swaps"
done 

