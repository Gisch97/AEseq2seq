-- TRAIN / TEST VIEWS

CREATE VIEW view_test_metrics AS 
with split_train_test AS(
SELECT 
	run_uuid,
    experiment_name,
	CASE WHEN p.train_file IS NULL THEN 'test' ELSE 'train' END AS TYPE
FROM view_params p
)
SELECT  
    s.experiment_name,
    m.run_uuid,
    m.name,
    m.test_loss ,
    m.test_Accuracy ,
    m.test_Accuracy_seq ,
    m.test_F1 
FROM view_metrics m 
JOIN split_train_test s ON s.run_uuid = m.run_uuid
WHERE s.TYPE ='test'
ORDER BY name;


CREATE VIEW view_train_metrics AS 
WITH split_train_test AS(
SELECT 
	run_uuid,
    experiment_name,
	CASE WHEN p.train_file IS NULL THEN 'test' ELSE 'train' END AS TYPE
FROM view_params p
)
SELECT  
    s.experiment_name,
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
JOIN split_train_test s ON s.run_uuid = m.run_uuid
WHERE s.TYPE ='train'
ORDER BY name;
 