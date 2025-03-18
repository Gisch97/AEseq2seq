-- view_params source

CREATE VIEW view_params AS
SELECT
    e.name AS experiment_name,
    r.run_uuid,
    r.name,
    r.lifecycle_stage,
    r.status,
    MAX(CASE WHEN p.key = 'arc_addition' THEN p.value END) AS arc_addition,
    MAX(CASE WHEN p.key = 'arc_embedding_dim' THEN p.value END) AS arc_embedding_dim,
    MAX(CASE WHEN p.key = 'arc_encoder_blocks' THEN p.value END) AS arc_encoder_blocks,
    MAX(CASE WHEN p.key = 'arc_features' THEN p.value END) AS arc_features,
    MAX(CASE WHEN p.key = 'arc_initial_volume' THEN p.value END) AS arc_initial_volume,
    MAX(CASE WHEN p.key = 'arc_latent_dim' THEN p.value END) AS arc_latent_dim,
    MAX(CASE WHEN p.key = 'arc_latent_volume' THEN p.value END) AS arc_latent_volume,
    MAX(CASE WHEN p.key = 'arc_num_conv' THEN p.value END) AS arc_num_conv,
    MAX(CASE WHEN p.key = 'arc_num_params' THEN p.value END) AS arc_num_params,
    MAX(CASE WHEN p.key = 'arc_pool_mode' THEN p.value END) AS arc_pool_mode,
    MAX(CASE WHEN p.key = 'arc_skip' THEN p.value END) AS arc_skip,
    MAX(CASE WHEN p.key = 'arc_up_mode' THEN p.value END) AS arc_up_mode,
    MAX(CASE WHEN p.key = 'device' THEN p.value END) AS device,
    MAX(CASE WHEN p.key = 'exp' THEN p.value END) AS exp,
    MAX(CASE WHEN p.key = 'global_config' THEN p.value END) AS global_config,
    MAX(CASE WHEN p.key = 'hyp_output_th' THEN p.value END) AS hyp_output_th,
    MAX(CASE WHEN p.key = 'hyp_scheduler' THEN p.value END) AS hyp_scheduler,
    MAX(CASE WHEN p.key = 'hyp_test_noise' THEN p.value END) AS hyp_test_noise,
    MAX(CASE WHEN p.key = 'max_epochs' THEN p.value END) AS max_epochs,
    MAX(CASE WHEN p.key = 'max_len' THEN p.value END) AS max_len,
    MAX(CASE WHEN p.key = 'max_length' THEN p.value END) AS max_length,
    MAX(CASE WHEN p.key = 'no_cache' THEN p.value END) AS no_cache,
    MAX(CASE WHEN p.key = 'nworkers' THEN p.value END) AS nworkers,
    MAX(CASE WHEN p.key = 'out_path' THEN p.value END) AS out_path,
    MAX(CASE WHEN p.key = 'run' THEN p.value END) AS run,
    MAX(CASE WHEN p.key = 'test_file' THEN p.value END) AS test_file,
    MAX(CASE WHEN p.key = 'train_file' THEN p.value END) AS train_file,
    MAX(CASE WHEN p.key = 'valid_file' THEN p.value END) AS valid_file,
    MAX(CASE WHEN p.key = 'valid_split' THEN p.value END) AS valid_split
FROM params p
LEFT JOIN runs r ON p.run_uuid = r.run_uuid
LEFT JOIN experiments e ON r.experiment_id = e.experiment_id
GROUP BY r.run_uuid, e.name;
