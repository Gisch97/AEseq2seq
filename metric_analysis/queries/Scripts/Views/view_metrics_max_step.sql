CREATE VIEW metrics_max_step AS
WITH max_steps AS (
    SELECT
        run_uuid,
        MAX(step) AS max_step
    FROM metrics
    GROUP BY run_uuid
)
SELECT
    m.run_uuid,
    m.step,
    MAX(CASE WHEN key = 'train_Accuracy' THEN value END) AS train_Accuracy,
    MAX(CASE WHEN key = 'train_Accuracy_seq' THEN value END) AS train_Accuracy_seq,
    MAX(CASE WHEN key = 'train_F1' THEN value END) AS train_F1,
    MAX(CASE WHEN key = 'train_Precision' THEN value END) AS train_Precision,
    MAX(CASE WHEN key = 'train_Recall' THEN value END) AS train_Recall,
    MAX(CASE WHEN key = 'train_ce_loss' THEN value END) AS train_ce_loss,
    MAX(CASE WHEN key = 'train_loss' THEN value END) AS train_loss,
    MAX(CASE WHEN key = 'train_loss_Accuracy' THEN value END) AS train_loss_Accuracy,
    MAX(CASE WHEN key = 'train_loss_Accuracy_seq' THEN value END) AS train_loss_Accuracy_seq,
    MAX(CASE WHEN key = 'train_loss_F1' THEN value END) AS train_loss_F1,
    MAX(CASE WHEN key = 'train_loss_Precision' THEN value END) AS train_loss_Precision,
    MAX(CASE WHEN key = 'train_loss_Recall' THEN value END) AS train_loss_Recall,
    MAX(CASE WHEN key = 'train_loss_ce_loss' THEN value END) AS train_loss_ce_loss,
    MAX(CASE WHEN key = 'train_loss_loss' THEN value END) AS train_loss_loss,
    MAX(CASE WHEN key = 'valid_Accuracy' THEN value END) AS valid_Accuracy,
    MAX(CASE WHEN key = 'valid_Accuracy_seq' THEN value END) AS valid_Accuracy_seq,
    MAX(CASE WHEN key = 'valid_F1' THEN value END) AS valid_F1,
    MAX(CASE WHEN key = 'valid_ce_loss' THEN value END) AS valid_ce_loss,
    MAX(CASE WHEN key = 'valid_loss' THEN value END) AS valid_loss,
    MAX(CASE WHEN key = 'valid_loss_Accuracy' THEN value END) AS valid_loss_Accuracy,
    MAX(CASE WHEN key = 'valid_loss_Accuracy_seq' THEN value END) AS valid_loss_Accuracy_seq,
    MAX(CASE WHEN key = 'valid_loss_F1' THEN value END) AS valid_loss_F1,
    MAX(CASE WHEN key = 'valid_loss_Precision' THEN value END) AS valid_loss_Precision,
    MAX(CASE WHEN key = 'valid_loss_Recall' THEN value END) AS valid_loss_Recall,
    MAX(CASE WHEN key = 'valid_loss_ce_loss' THEN value END) AS valid_loss_ce_loss,
    MAX(CASE WHEN key = 'valid_loss_loss' THEN value END) AS valid_loss_loss
FROM metrics m
INNER JOIN max_steps ms ON m.run_uuid = ms.run_uuid AND m.step = ms.max_step
GROUP BY m.run_uuid, m.step;