"""jobs nullable platform fields

Revision ID: 3f8c2a1d9e47
Revises: 1ad06d16b45b
Create Date: 2026-05-02

"""
from typing import Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '3f8c2a1d9e47'
down_revision: Union[str, None] = '1ad06d16b45b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ALTER COLUMN organization_id DROP NOT NULL")
    op.execute("ALTER TABLE jobs ALTER COLUMN project_id DROP NOT NULL")
    op.execute("ALTER TABLE jobs ALTER COLUMN uploaded_by_user_id DROP NOT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs ALTER COLUMN organization_id SET NOT NULL")
    op.execute("ALTER TABLE jobs ALTER COLUMN project_id SET NOT NULL")
    op.execute("ALTER TABLE jobs ALTER COLUMN uploaded_by_user_id SET NOT NULL")
