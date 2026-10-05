"""Persist incident workflow and event timelines."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "3c4d5e6f7a8b"
down_revision: Union[str, None] = "2b3c4d5e6f7a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.create_table(
        "incidents",
        sa.Column("id", sa.String(length=80), primary_key=True),
        sa.Column("icao24", sa.String(length=6), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="new"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("rule_flags", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("comments", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("timeline", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("source", sa.String(length=80), nullable=False, server_default="unknown"),
    )
    op.create_index(op.f("ix_incidents_icao24"), "incidents", ["icao24"], unique=False)
    op.create_index(op.f("ix_incidents_created_at"), "incidents", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_incidents_created_at"), table_name="incidents")
    op.drop_index(op.f("ix_incidents_icao24"), table_name="incidents")
    op.drop_table("incidents")
