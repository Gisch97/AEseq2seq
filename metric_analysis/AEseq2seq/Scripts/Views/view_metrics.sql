-- view_metrics source

CREATE VIEW view_metrics AS
SELECT
    m.run_uuid,
    r.name,
    step,
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
    MAX(CASE WHEN key = 'valid_loss_loss' THEN value END) AS valid_loss_loss,
    MAX(CASE WHEN key = 'test_Accuracy' THEN value END) AS test_Accuracy,
    MAX(CASE WHEN key = 'test_Accuracy_seq' THEN value END) AS test_Accuracy_seq,
    MAX(CASE WHEN key = 'test_F1' THEN value END) AS test_F1,
    MAX(CASE WHEN key = 'test_ce_loss' THEN value END) AS test_ce_loss,
    MAX(CASE WHEN key = 'test_loss' THEN value END) AS test_loss 
FROM metrics m
JOIN runs r ON r.run_uuid = m.run_uuid
GROUP BY m.run_uuid, step;

