#!/bin/bash

### EXPERIMENTO Y MODELOS SELECCIONADOS
MODEL_NAME="unet_original" 
MODEL_PATH="src/seq2seq/models/paper_based"
INIT="src/seq2seq/__init__.py"
GLOBAL_CONFIG="config/global.json" 
MODEL_FILE="$MODEL_PATH/$MODEL_NAME.py"


# Hyperparameters
MAX_EPOCHS=20
pool_mode=('avg' 'max')
up_mode=('transpose' 'upsample')
addition=('sum' 'cat')
skips=(0 1) 

n_4=(0 1 2 3)
n_8=(0 1 2)

EXPERIMENT_NAME="Unet_fixed_models_ablation"
BASE_OUTPUT_PATH="results/UNet_selection/$EXPERIMENT_NAME"
mkdir -p "$BASE_OUTPUT_PATH"

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

for nc_4 in "${n_4[@]}"; do
    for nc_8 in "${n_8[@]}"; do
        for pool in "${pool_mode[@]}"; do
            for up in "${up_mode[@]}"; do
                for add in "${addition[@]}"; do
                    for sk in "${skips[@]}";do
                        # Se configura el valor de skip fijo en 0 para este bloque (se puede anidar otro bucle si se desea variar)
                        base_name="model_"
                        save_name="${base_name}n4-$nc_4-n8-$nc_8-p-$pool-up-$up-add-$add-skip$sk"
                        echo "Ejecutando:$save_name"
                        # # Modificar la configuración del modelo para pool, up y skip
                        sed -i \
                            -e "82s/pool_mode=['\"][^'\"]*['\"]/pool_mode=\"$pool\"/" \
                            -e "83s/up_mode=['\"][^'\"]*['\"]/up_mode=\"$up\"/" \
                            -e "84s/skip=[0-9e.-]*/skip=$sk/" \
                            -e "85s/addition=['\"][^'\"]*['\"]/addition=\"$add\"/" \
                            -e "91s/n_4=[0-9e.-]*/n_4=$nc_4/" \
                            -e "92s/n_8=[0-9e.-]*/n_8=$nc_8/" \
                            "$MODEL_FILE"
                        bash scripts/train_test.sh "$BASE_OUTPUT_PATH" "$save_name" 
                    done
                done
            done
        done 
    done
done  

