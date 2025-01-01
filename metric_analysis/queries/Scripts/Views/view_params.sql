-- view_params source

CREATE VIEW view_params AS
SELECT
    r.run_uuid,
    r.name,
    e.name AS experiment_name,
    MAX(CASE WHEN p.key = 'arc_dilation_resnet1d' THEN p.value END) AS arc_dilation_resnet1d,
    MAX(CASE WHEN p.key = 'arc_embedding_dim' THEN p.value END) AS arc_embedding_dim,
    MAX(CASE WHEN p.key = 'arc_filters' THEN p.value END) AS arc_filters,
    MAX(CASE WHEN p.key = 'arc_kernel' THEN p.value END) AS arc_kernel,
    MAX(CASE WHEN p.key = 'arc_latent_dim' THEN p.value END) AS arc_latent_dim,
    MAX(CASE WHEN p.key = 'arc_num_layers' THEN p.value END) AS arc_num_layers,
    MAX(CASE WHEN p.key = 'arc_rank' THEN p.value END) AS arc_rank,
    MAX(CASE WHEN p.key = 'arc_resnet_bottleneck_factor' THEN p.value END) AS arc_resnet_bottleneck_factor,
    MAX(CASE WHEN p.key = 'batch' THEN p.value END) AS batch,
    MAX(CASE WHEN p.key = 'batch_size' THEN p.value END) AS batch_size,
    MAX(CASE WHEN p.key = 'cache_path' THEN p.value END) AS cache_path,
    MAX(CASE WHEN p.key = 'command' THEN p.value END) AS command,
    MAX(CASE WHEN p.key = 'd' THEN p.value END) AS d,
    MAX(CASE WHEN p.key = 'device' THEN p.value END) AS device,
    MAX(CASE WHEN p.key = 'exp' THEN p.value END) AS exp,
    MAX(CASE WHEN p.key = 'global_config' THEN p.value END) AS global_config,
    MAX(CASE WHEN p.key = 'hyp_device' THEN p.value END) AS hyp_device,
    MAX(CASE WHEN p.key = 'hyp_embedding_dim' THEN p.value END) AS hyp_embedding_dim,
    MAX(CASE WHEN p.key = 'hyp_interaction_prior' THEN p.value END) AS hyp_interaction_prior,
    MAX(CASE WHEN p.key = 'hyp_lr' THEN p.value END) AS hyp_lr,
    MAX(CASE WHEN p.key = 'hyp_negative_weight' THEN p.value END) AS hyp_negative_weight,
    MAX(CASE WHEN p.key = 'hyp_output_th' THEN p.value END) AS hyp_output_th,
    MAX(CASE WHEN p.key = 'hyp_scheduler' THEN p.value END) AS hyp_scheduler,
    MAX(CASE WHEN p.key = 'hyp_verbose' THEN p.value END) AS hyp_verbose,
    MAX(CASE WHEN p.key = 'j' THEN p.value END) AS j,
    MAX(CASE WHEN p.key = 'max_epochs' THEN p.value END) AS max_epochs,
    MAX(CASE WHEN p.key = 'max_len' THEN p.value END) AS max_len,
    MAX(CASE WHEN p.key = 'max_length' THEN p.value END) AS max_length,
    MAX(CASE WHEN p.key = 'no_cache' THEN p.value END) AS no_cache,
    MAX(CASE WHEN p.key = 'nworkers' THEN p.value END) AS nworkers,
    MAX(CASE WHEN p.key = 'out_path' THEN p.value END) AS out_path,
    MAX(CASE WHEN p.key = 'quiet' THEN p.value END) AS quiet,
    MAX(CASE WHEN p.key = 'run' THEN p.value END) AS run,
    MAX(CASE WHEN p.key = 'train_file' THEN p.value END) AS train_file,
    MAX(CASE WHEN p.key = 'valid_file' THEN p.value END) AS valid_file,
    MAX(CASE WHEN p.key = 'valid_split' THEN p.value END) AS valid_split,
    MAX(CASE WHEN p.key = 'verbose' THEN p.value END) AS verbose
FROM params p
LEFT JOIN runs r ON p.run_uuid = r.run_uuid
LEFT JOIN experiments e ON r.experiment_id = e.experiment_id
GROUP BY r.run_uuid, e.name;