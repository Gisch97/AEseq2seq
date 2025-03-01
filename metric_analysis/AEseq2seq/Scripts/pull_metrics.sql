-- Check experiments ------------------------------------------------------------------------------------
SELECT * FROM experiments

-- PULL TRAIN  ------------------------------------------------------------------------------------
WITH run_set AS (
    SELECT  
        run_uuid
    FROM runs 
    WHERE experiment_id == 27
      AND lifecycle_stage <> 'deleted'
      AND status = 'FINISHED'
)
SELECT tm.* 
FROM view_train_metrics tm
JOIN run_set r ON tm.run_uuid = r.run_uuid

-- PULL TEST ------------------------------------------------------------------------------------
WITH run_set AS (
    SELECT  
        run_uuid
    FROM runs 
    WHERE experiment_id == 19
      AND lifecycle_stage <> 'deleted'
      AND status = 'FINISHED'
)
SELECT tm.* 
FROM view_test_metrics tm
JOIN run_set r ON tm.run_uuid = r.run_uuid

-- check run ---------------------------------------------------------------------------------------------------
SELECT * FROM runs 
WHERE experiment_id == 11
AND lifecycle_stage <> 'deleted'
AND status = 'FINISHED'
