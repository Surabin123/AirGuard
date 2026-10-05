"""Persist transponder squawk codes for emergency monitoring."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "2b3c4d5e6f7a"
down_revision: Union[str, None] = "1a2b3c4d5e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column("aircraft_states", sa.Column("squawk", sa.String(length=4), nullable=True))
    op.alter_column("aircraft_states", "source", existing_type=sa.String(length=20), type_=sa.String(length=80), existing_nullable=False)


def downgrade() -> None:
    op.alter_column("aircraft_states", "source", existing_type=sa.String(length=80), type_=sa.String(length=20), existing_nullable=False)
    op.drop_column("aircraft_states", "squawk")
