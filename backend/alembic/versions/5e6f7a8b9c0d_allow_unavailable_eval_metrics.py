"""Allow metrics to be unavailable when labels do not support them."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "5e6f7a8b9c0d"
down_revision: Union[str, None] = "4d5e6f7a8b9c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("model_runs", "false_positives", existing_type=sa.Integer(), nullable=True)
    op.alter_column("model_runs", "true_negatives", existing_type=sa.Integer(), nullable=True)
    op.alter_column("model_runs", "precision", existing_type=sa.Float(), nullable=True)
    op.alter_column("model_runs", "recall", existing_type=sa.Float(), nullable=True)
    op.alter_column("model_runs", "f1", existing_type=sa.Float(), nullable=True)


def downgrade() -> None:
    # The older schema cannot represent unavailable values; preserve downgrade
    # compatibility with explicit zero placeholders only when rolling back.
    op.execute("UPDATE model_runs SET false_positives = 0 WHERE false_positives IS NULL")
    op.execute("UPDATE model_runs SET true_negatives = 0 WHERE true_negatives IS NULL")
    op.execute("UPDATE model_runs SET precision = 0 WHERE precision IS NULL")
    op.execute("UPDATE model_runs SET recall = 0 WHERE recall IS NULL")
    op.execute("UPDATE model_runs SET f1 = 0 WHERE f1 IS NULL")
    op.alter_column("model_runs", "f1", existing_type=sa.Float(), nullable=False)
    op.alter_column("model_runs", "recall", existing_type=sa.Float(), nullable=False)
    op.alter_column("model_runs", "precision", existing_type=sa.Float(), nullable=False)
    op.alter_column("model_runs", "true_negatives", existing_type=sa.Integer(), nullable=False)
    op.alter_column("model_runs", "false_positives", existing_type=sa.Integer(), nullable=False)
