"""drop parser_stage_log_id from streets_raw

Revision ID: a1b2c3d4e5f6
Revises: 3f8c2a1d9e47
Create Date: 2026-05-03

"""
from typing import Union
from alembic import op

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '3f8c2a1d9e47'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE streets_raw DROP COLUMN IF EXISTS parser_stage_log_id")


def downgrade() -> None:
    op.execute("ALTER TABLE streets_raw ADD COLUMN IF NOT EXISTS parser_stage_log_id UUID")
