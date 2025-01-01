import pandas as pd
import sqlite3

# Conectar a la base de datos SQLite
DATABASE_PATH = "mlruns.db"
conn = sqlite3.connect(DATABASE_PATH)

# Consulta SQL para pivotear la tabla directamente
query = """
SELECT 
    r.run_uuid,
    r.status,
    r.experiment_id,
    r.name AS run_name,
    m.step,
    MAX(CASE WHEN m.key = 'train_loss' THEN m.value END) AS train_loss,
    MAX(CASE WHEN m.key = 'train_loss_loss' THEN m.value END) AS train_loss_loss,
    MAX(CASE WHEN m.key = 'train_ce_loss' THEN m.value END) AS train_ce_loss,
    MAX(CASE WHEN m.key = 'train_loss_ce_loss' THEN m.value END) AS train_loss_ce_loss,
    MAX(CASE WHEN m.key = 'train_F1' THEN m.value END) AS train_F1
FROM 
    metrics m
JOIN 
    runs r ON m.run_uuid = r.run_uuid
GROUP BY 
    r.run_uuid, r.status, r.experiment_id, r.name, m.step
"""

# Ejecutar la consulta y cargar los resultados en un DataFrame
df_pivot_sql = pd.read_sql_query(query, conn)

# Cerrar la conexión
conn.close()

# Visualizar la tabla resultante
print(df_pivot_sql.head())
