SELECT * FROM runs 
WHERE experiment_id == 11
AND lifecycle_stage <> 'deleted'
AND status = 'FINISHED'

WITH run_set AS (
    SELECT  
        run_uuid, 
        name,			
        (end_time - start_time) / 1000 AS time_seconds
    FROM runs 
    WHERE experiment_id == 11
      AND lifecycle_stage <> 'deleted'
      AND status = 'FINISHED'
) 
SELECT 
	CASE WHEN p.train_file IS NULL THEN 'test' ELSE 'train' END AS TYPE,
    rs.time_seconds,
    p.*
FROM view_params p
JOIN run_set rs ON p.run_uuid = rs.run_uuid
ORDER BY experiment_name, TYPE, name;

SELECT t.*, r.experiment_id
FROM tags t
JOIN runs r ON t.run_uuid = r.run_uuid
WHERE t.key LIKE '%command%';


SELECT * FROM view_params