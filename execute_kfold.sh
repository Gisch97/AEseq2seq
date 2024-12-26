#!/bin/bash
set -e

# Directorios principales
DATA_DIR="data/ArchiveII-KFold/fam-similarity"
OUT_DIR="results/plain_kfold_similarity"
LOG_FILE="$OUT_DIR/kfold_log.txt"

mkdir -p "$OUT_DIR"

# Función para actualizar archivos JSON
update_json_files() {
    local fold=$1
    local config_dir="config"  # Directorio donde están los archivos JSON

    # Actualizar global.json usando sed
    sed -i "s/\"run\": \"fam_sim_fold_[0-9]\+\"/\"run\": \"fam_sim_fold_${fold}\"/" "$config_dir/global.json"

    # Actualizar train.json
    cat > "$config_dir/train.json" << EOF
{
    "train_file": "data/ArchiveII-KFold/fam-similarity/train_${fold}.csv",
    "valid_file": "data/ArchiveII-KFold/fam-similarity/valid_${fold}.csv",
    "out_path": "results/ArchiveII-KFold/fam-similarity/fold_${fold}"
}
EOF

    # Actualizar test.json
    cat > "$config_dir/test.json" << EOF
{
    "test_file": "data/ArchiveII-KFold/fam-similarity/test_${fold}.csv",
    "model_weights": "results/plain_latent32/lr1e-3-ot0.5/weights.pmt",
    "out_path": "results/ArchiveII-KFold/fam-similarity/fold_${fold}/testlog_${fold}.csv"
}
EOF
}

run_fold() {
    local fold=$1
    TRAIN_FILE="$DATA_DIR/train_${fold}.csv"
    VAL_FILE="$DATA_DIR/valid_${fold}.csv"
    OUTPUT_DIR="$DATA_DIR/fold_${fold}_results"

    mkdir -p "$OUTPUT_DIR"
    echo "Ejecutando Fold $fold..." >> "$LOG_FILE"

    # Actualizar archivos JSON antes de la ejecución
    update_json_files "$fold"

    # Ejecutar el entrenamiento
    seq2seq train 
}

KFOLDS=5

for fold in $(seq 0 $(($KFOLDS - 1))); do
    echo "Iniciando fold $fold..."
    run_fold "$fold"
done

echo "K-Fold completado. Revisa $LOG_FILE para más detalles."