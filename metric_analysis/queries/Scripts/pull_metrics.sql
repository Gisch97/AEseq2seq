-- PULL TRAIN
SELECT * FROM runs 
WHERE experiment_id IN (7,9)
AND lifecycle_stage <> 'deleted'
AND status = 'FINISHED'

WITH run_set AS (
    SELECT  
        run_uuid
    FROM runs 
    WHERE experiment_id IN (7, 9)
      AND lifecycle_stage <> 'deleted'
      AND status = 'FINISHED'
),
split_train_test AS(
SELECT 
	run_uuid,
	CASE WHEN p.train_file IS NULL THEN 'test' ELSE 'train' END AS TYPE
FROM view_params p
)
SELECT  
    m.run_uuid,
    m.name,
    m.step,
    m.train_loss ,
    m.train_Accuracy ,
    m.train_F1,
    m.valid_loss ,
    m.valid_Accuracy,
    m.valid_Accuracy_seq ,
    m.valid_F1
FROM view_metrics m
JOIN run_set rs ON m.run_uuid = rs.run_uuid
JOIN split_train_test s ON s.run_uuid = m.run_uuid
WHERE s.TYPE ='train'
ORDER BY name;



-- PULL TEST
SELECT * FROM runs 
WHERE experiment_id IN (7,9)
AND lifecycle_stage <> 'deleted'
AND status = 'FINISHED'

WITH run_set AS (
    SELECT  
        run_uuid
    FROM runs 
    WHERE experiment_id IN (7, 9)
      AND lifecycle_stage <> 'deleted'
      AND status = 'FINISHED'
),
split_train_test AS(
SELECT 
	run_uuid,
	CASE WHEN p.train_file IS NULL THEN 'test' ELSE 'train' END AS TYPE
FROM view_params p
)
SELECT  
    m.run_uuid,
    m.name,
    m.test_loss ,
    m.test_Accuracy ,
    m.test_Accuracy_seq ,
    m.test_F1 
FROM view_metrics m
JOIN run_set rs ON m.run_uuid = rs.run_uuid
JOIN split_train_test s ON s.run_uuid = m.run_uuid
WHERE s.TYPE ='test'
ORDER BY name;
