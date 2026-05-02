"""streets_raw_work_type_to_array

Revision ID: 1ad06d16b45b
Revises: 27d719695777
Create Date: 2026-05-02 17:11:53.096494

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1ad06d16b45b'
down_revision: Union[str, None] = '27d719695777'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE streets_raw
        RENAME COLUMN work_type TO work_types
    """)
    op.execute("""
        ALTER TABLE streets_raw
        ALTER COLUMN work_types TYPE TEXT[]
        USING ARRAY[work_types]::TEXT[]
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE streets_raw
        ALTER COLUMN work_types TYPE TEXT
        USING work_types[1]
    """)
    op.execute("""
        ALTER TABLE streets_raw
        RENAME COLUMN work_types TO work_type
    """)
