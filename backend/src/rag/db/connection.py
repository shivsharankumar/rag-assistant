"""Postgres connection helper."""
from contextlib import contextmanager
from collections.abc import Iterator
import psycopg
from src.rag.config import settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """
    Yield a Postgres connection, closing it after use.
    
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    conn = psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )
    try:
        yield conn
    finally:
        conn.close()