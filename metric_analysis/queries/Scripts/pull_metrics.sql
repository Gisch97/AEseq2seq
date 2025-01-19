-- PULL TRAIN
SELECT * FROM runs 
WHERE experiment_id == 11
AND lifecycle_stage <> 'deleted'
AND status = 'FINISHED'

SELECT * FROM runs

WITH run_set AS (
    SELECT  
        run_uuid
    FROM runs 
    WHERE experiment_id == 11
      AND lifecycle_stage <> 'deleted'
      AND status = 'FINISHED'
)
SELECT tm.* 
FROM view_train_metrics tm
JOIN run_set r ON tm.run_uuid = r.run_uuid


WITH run_set AS (
    SELECT  
        run_uuid
    FROM runs 
    WHERE experiment_id == 11
      AND lifecycle_stage <> 'deleted'
      AND status = 'FINISHED'
)
SELECT tm.* 
FROM view_test_metrics tm
JOIN run_set r ON tm.run_uuid = r.run_uuid

