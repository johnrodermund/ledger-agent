-- Roles are cluster-wide. Table grants are applied per database by
-- ledger_agent.db.grant_roles, so neither role owns any table.
-- Passwords are test values matching .env.example.
CREATE ROLE ledger_ro LOGIN PASSWORD 'ro_test_pw';
CREATE ROLE ledger_rw LOGIN PASSWORD 'rw_test_pw';
