CREATE VIEW view_metrics_best_epoch AS
WITH data AS (
  -- Primero obtenemos el último/máximo best_epoch para cada run_uuid
  WITH max_epochs AS (
    SELECT 
      run_uuid,
      MAX(CAST(value AS int)) as final_best_epoch
    FROM metrics
    WHERE key = 'best_epoch'
    GROUP BY run_uuid
  ),
  -- Luego pivotamos todas las métricas
  metrics_pivot AS (
    SELECT
    m.run_uuid, 
    step,
    MAX(CASE WHEN key = 'train_Accuracy' THEN value END) AS train_Accuracy,
    MAX(CASE WHEN key = 'train_Accuracy_seq' THEN value END) AS train_Accuracy_seq,
    MAX(CASE WHEN key = 'train_F1' THEN value END) AS train_F1, 
    MAX(CASE WHEN key = 'train_ce_loss' THEN value END) AS train_ce_loss,
    MAX(CASE WHEN key = 'train_loss' THEN value END) AS train_loss,
    MAX(CASE WHEN key = 'valid_Accuracy' THEN value END) AS valid_Accuracy,
    MAX(CASE WHEN key = 'valid_Accuracy_seq' THEN value END) AS valid_Accuracy_seq,
    MAX(CASE WHEN key = 'valid_F1' THEN value END) AS valid_F1,
    MAX(CASE WHEN key = 'valid_ce_loss' THEN value END) AS valid_ce_loss,
    MAX(CASE WHEN key = 'valid_loss' THEN value END) AS valid_loss
    FROM metrics m
    GROUP BY m.run_uuid, m.step
  )
  SELECT 
    mp.*
  FROM metrics_pivot mp
  JOIN max_epochs me ON mp.run_uuid = me.run_uuid 
  WHERE mp.step = me.final_best_epoch
)
SELECT * FROM data;