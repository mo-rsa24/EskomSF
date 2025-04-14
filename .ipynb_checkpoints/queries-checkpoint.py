def get_sample_rows(table_name, limit=5):
    query = f"SELECT TOP {limit} * FROM {table_name}"
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
    except pyodbc.Error as e:
        print("Query failed:", e)
        return []
