#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
MODEL_NAME="unet_original" 
MODEL_PATH="src/seq2seq/models/paper_based"
INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"


# Hyperparameters
MAX_EPOCHS=20
pool_mode=('avg')
up_mode=('transpose')
addition=('sum')
skip=(0 1) 

n_4=(3)
n_8=(0 1)

p=(0.0 0.03125 0.0625 0.09375 0.125 0.15625 0.1875 0.21875 0.25 0.28125 0.3125 0.34375 0.375 0.40625 0.4375 0.46875 0.5 0.53125 0.5625 0.59375 0.625 0.65625 0.6875 0.71875 0.75 0.78125 0.8125 0.84375 0.875 0.90625 0.9375 0.96875 1.0)

EXPERIMENT_NAME="Unet_nc4_nc8_noise_0_1"
BASE_OUTPUT_PATH="results/UNet_selection/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"
for nc_4 in "${n_4[@]}"; do
    for nc_8 in "${n_8[@]}"; do
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

# Bucle para iterar sobre las configuraciones de pool, up y addition 
        # Se configura el valor de skip fijo en 0 para este bloque (se puede anidar otro bucle si se desea variar)
        base_name="n4-$nc_4-n8-$nc_8"
        echo "Ejecutando: n4-$nc_4-n8-$nc_8 skip=0"
        save_name="${base_name}-skip0"
        # # Modificar la configuración del modelo para pool, up y skip
        sed -i \
            -e "87s/\(skip=\)[0-9e.-]*/skip=0/" \
            -e "94s/n_4=[0-9e.-]*/n_4=$nc_4/" \
            -e "95s/n_8=[0-9e.-]*/n_8=$nc_8/" \
            "$MODEL_FILE"

        bash scripts/train.sh "$BASE_OUTPUT_PATH" "$save_name" 
        for swaps in ${p[@]}; do
            bash scripts/test_w_swap.sh "$BASE_OUTPUT_PATH" "$save_name" "$swaps"
        done

        echo "Ejecutando: n4-$nc_4-n8-$nc_8 skip=1"
        # Actualizar el parámetro addition en el modelo
        sed -i \
            -e "87s/\(skip=\)[0-9e.-]*/skip=1/" \
            "$MODEL_FILE"
        # Definir el nombre de guardado usando base_name y el valor de addition
        save_name="$base_name-skip1"
        echo "save_name: $save_name"

        bash scripts/train.sh "$BASE_OUTPUT_PATH" "$save_name" 
        for swaps in ${p[@]}; do
            bash scripts/test_w_swap.sh "$BASE_OUTPUT_PATH" "$save_name" "$swaps"
        done
    done
done  

