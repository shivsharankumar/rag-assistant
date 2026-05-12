import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ragdb",
    user="postgres",
    password="devpass",
)

with conn.cursor() as cur:
    cur.execute("SELECT version()")
    print(cur.fetchone())

conn.close()