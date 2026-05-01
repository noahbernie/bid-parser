# Database connection module — asyncpg client for AWS RDS Postgres
import os
import asyncpg


async def get_db_connection() -> asyncpg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return await asyncpg.connect(url)


async def init_db() -> None:
    try:
        conn = await get_db_connection()
        await conn.fetchval("SELECT 1")
        await conn.close()
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")
