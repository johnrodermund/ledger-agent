import uuid

import pytest

from ledger_agent.db import connect_admin, grant_roles


def pytest_configure(config):
    # Raised by langchain-core when langgraph imports its checkpoint serializer. The
    # audit keeps no checkpoints, so nothing here deserializes through that path.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The default value of `allowed_objects` will change"
        ":langchain_core._api.deprecation.LangChainPendingDeprecationWarning")


@pytest.fixture
def scratch_db():
    name = f"scratch_{uuid.uuid4().hex[:12]}"
    with connect_admin("postgres") as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    try:
        with connect_admin(name) as c:
            c.execute("CREATE TABLE probe (id integer)")
            grant_roles(c)
        yield name
    finally:
        with connect_admin("postgres") as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
