WITH parameters AS (
SELECT  p.*
FROM view_params p
JOIN view_metrics_best_epoch be ON p.run_uuid = be.run_uuid 
WHERE experiment_name = 'paper_based_unet_fam_split' 
AND p.lifecycle_stage <> 'deleted'
AND p.status = 'FINISHED'
)
SELECT	p.*,
		m.step AS best_epoch,
		m.train_loss,
		m.train_Accuracy,
		m.train_Accuracy_seq,
		m.train_F1,
		m.valid_loss,
		m.valid_Accuracy,
		m.valid_Accuracy_seq,
		m.valid_F1,
		t.test_loss,
		t.test_Accuracy,
		t.test_Accuracy_seq,
		t.test_F1
FROM parameters p
LEFT JOIN view_metrics_best_epoch m ON p.run_uuid = m.run_uuid 
LEFT JOIN view_test_metrics t ON t.name = p.name AND p.experiment_name = t.experiment_name
WHERE m.step < 20
ORDER BY name


SELECT * FROM view_metrics_best_epoch


SELECT * FROM metrics m 
WHERE m.key = 'best_epoch'