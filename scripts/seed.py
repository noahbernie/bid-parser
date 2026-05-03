"""
Seed script — inserts baseline test data into the database.
Safe to run multiple times (skips rows that already exist).

Usage:
    DATABASE_URL=postgresql://postgres@localhost:5432/blueroads_dev python scripts/seed.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def seed():
    import asyncpg
    from dotenv import load_dotenv
    load_dotenv()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = await asyncpg.connect(db_url)
    print(f"Connected to {db_url.split('@')[-1]}\n")

    try:
        # --- Organization ---
        org = await conn.fetchrow("SELECT id FROM organizations WHERE name = $1", "VSS International")
        if org:
            org_id = org["id"]
            print(f"✓ Org already exists: {org_id}")
        else:
            org_id = await conn.fetchval("""
                INSERT INTO organizations (name, subscription_status)
                VALUES ($1, 'active')
                RETURNING id
            """, "VSS International")
            print(f"✓ Org created: {org_id}")

        # --- User ---
        user = await conn.fetchrow("SELECT id FROM users WHERE email = $1", "nick@vssinternational.com")
        if user:
            user_id = user["id"]
            print(f"✓ User already exists: {user_id}")
        else:
            user_id = await conn.fetchval("""
                INSERT INTO users (organization_id, email, name, role)
                VALUES ($1, $2, 'Nick Roberts', 'org_admin')
                RETURNING id
            """, org_id, "nick@vssinternational.com")
            print(f"✓ User created: {user_id}")

        # --- Project ---
        project = await conn.fetchrow(
            "SELECT id FROM projects WHERE organization_id = $1 AND name = $2",
            org_id, "Test Bid Project"
        )
        if project:
            project_id = project["id"]
            print(f"✓ Project already exists: {project_id}")
        else:
            project_id = await conn.fetchval("""
                INSERT INTO projects (organization_id, created_by_user_id, name, location, status)
                VALUES ($1, $2, 'Test Bid Project', 'San Diego, CA', 'in_progress')
                RETURNING id
            """, org_id, user_id)
            print(f"✓ Project created: {project_id}")

        # --- Job ---
        job = await conn.fetchrow(
            "SELECT id FROM jobs WHERE project_id = $1 AND job_name = $2",
            project_id, "test_bid.pdf"
        )
        if job:
            job_id = job["id"]
            print(f"✓ Job already exists: {job_id}")
        else:
            job_id = await conn.fetchval("""
                INSERT INTO jobs (project_id, organization_id, uploaded_by_user_id, job_name, status)
                VALUES ($1, $2, $3, 'test_bid.pdf', 'uploaded')
                RETURNING id
            """, project_id, org_id, user_id)
            print(f"✓ Job created: {job_id}")

        print(f"""
Seed complete. IDs for testing:
  org_id:     {org_id}
  user_id:    {user_id}
  project_id: {project_id}
  job_id:     {job_id}
""")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())
