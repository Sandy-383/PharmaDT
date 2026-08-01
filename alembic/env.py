"""Alembic environment.

The database URL comes from :mod:`pharmadt.config`, not from ``alembic.ini``, so
migrations and the application can never disagree about which database they are
talking to. The placeholder in ``alembic.ini`` is left unused on purpose.
"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context
from pharmadt.config import settings
from pharmadt.core.db import Base

# Importing the models module is what populates Base.metadata. Without it
# autogenerate would compare against an empty schema and cheerfully emit a
# migration that drops every table.
import pharmadt.core.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = create_engine(settings.database_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Without these, autogenerate ignores column type changes and
            # server-side defaults — the two edits most likely to be missed.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
