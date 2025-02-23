SELECT  p.run_uuid,
		p.name,
		be.step AS best_epoch,
		p.arc_filters,
		p.arc_rank,
		p.arc_kernel,
		p.arc_stride_1,
		p.arc_stride_2,
		CASE
			WHEN p.name LIKE '%no-skips%' THEN 0
			ELSE 1 
		END AS arc_skip_conn,
		p.hyp_lr,
		p.hyp_output_th,
		p.hyp_scheduler
FROM view_params p
JOIN view_metrics_best_epoch be ON p.run_uuid = be.run_uuid 
WHERE experiment_name == 'Unet'

SELECT  p.name,
		p.arc_filters,
		p.arc_rank,
		p.arc_kernel,
		p.arc_stride_1,
		p.arc_stride_2,
		CASE
			WHEN p.name LIKE '%no-skips%' THEN 0
			ELSE 1 
		END AS arc_skip_conn,
		p.hyp_lr,
		p.hyp_output_th,
		p.hyp_scheduler
FROM view_params p
WHERE experiment_name == 'Unet'
AND command IS NULL 





