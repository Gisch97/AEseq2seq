-- Archivo SQL con consultas útiles para MLFlow

-- 1. Entrenamientos finalizados por experimento
SELECT 
    experiment_id, 
    COUNT(*) AS num_runs
FROM 
    runs
WHERE 
    status = 'FINISHED'
GROUP BY 
    experiment_id
ORDER BY 
    num_runs DESC;

-- 2. Entrenamientos más recientes
SELECT 
    run_uuid, 
    name AS run_name, 
    start_time, 
    status
FROM 
    runs
ORDER BY 
    start_time DESC
LIMIT 5;

-- 3. Mejor métrica para cada entrenamiento
SELECT 
    r.run_uuid, 
    r.name AS run_name, 
    MIN(m.value) AS min_train_loss
FROM 
    metrics m
JOIN 
    runs r ON m.run_uuid = r.run_uuid
WHERE 
    m.key = 'train_loss'
GROUP BY 
    r.run_uuid, r.name;

-- 4. Entrenamientos con alta pérdida
SELECT 
    r.run_uuid, 
    r.name AS run_name, 
    m.value AS train_loss
FROM 
    metrics m
JOIN 
    runs r ON m.run_uuid = r.run_uuid
WHERE 
    m.key = 'train_loss' 
    AND m.value > 1.0;

-- 5. Comparación de parámetros entre entrenamientos
SELECT 
    r.run_uuid, 
    r.name AS run_name, 
    p.key AS param_key, 
    p.value AS param_value
FROM 
    params p
JOIN 
    runs r ON p.run_uuid = r.run_uuid
WHERE 
    r.experiment_id = 1
ORDER BY 
    r.run_uuid, p.key;

-- 6. Progreso de una métrica por pasos
SELECT 
    step, 
    value AS train_loss
FROM 
    metrics
WHERE 
    run_uuid = '0360d77ee3cc4bbf96de3886c7ee868a'
    AND key = 'train_loss'
ORDER BY 
    step ASC;

-- 7. Entrenamientos por estado
SELECT 
    status, 
    COUNT(*) AS num_runs
FROM 
    runs
GROUP BY 
    status;

-- 8. Comparación de experimentos
SELECT 
    r.experiment_id, 
    AVG(m.value) AS avg_valid_loss
FROM 
    metrics m
JOIN 
    runs r ON m.run_uuid = r.run_uuid
WHERE 
    m.key = 'valid_loss_loss'
GROUP BY 
    r.experiment_id
ORDER BY 
    avg_valid_loss ASC;

-- 9. Entrenamientos con mejores F1-scores
SELECT 
    r.run_uuid, 
    r.name AS run_name, 
    m.value AS train_F1
FROM 
    metrics m
JOIN 
    runs r ON m.run_uuid = r.run_uuid
WHERE 
    m.key = 'train_F1'
ORDER BY 
    train_F1 DESC
LIMIT 10;

-- 10. Duración de cada entrenamiento
SELECT 
    run_uuid, 
    name AS run_name, 
    (end_time - start_time) AS duration
FROM 
    runs
WHERE 
    status = 'FINISHED'
ORDER BY 
    duration DESC;

-- 11. Entrenamientos con métricas faltantes
SELECT 
    r.run_uuid, 
    r.name AS run_name
FROM 
    runs r
LEFT JOIN 
    metrics m ON r.run_uuid = m.run_uuid AND m.key = 'train_loss'
WHERE 
    m.value IS NULL;

-- 12. Tendencias de métricas por experimentos
SELECT 
    r.experiment_id, 
    AVG(m.value) AS avg_train_loss, 
    r.start_time
FROM 
    metrics m
JOIN 
    runs r ON m.run_uuid = r.run_uuid
WHERE 
    m.key = 'train_loss'
GROUP BY 
    r.experiment_id, r.start_time
ORDER BY 
    r.start_time ASC;
