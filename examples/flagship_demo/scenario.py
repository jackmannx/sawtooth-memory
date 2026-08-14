"""
scenario.py — A canned, realistic long-running coding-agent session.

Critical facts are planted early (production DB connection ID, incident
ticket, file path, commit hash). ~35 turns of ordinary debugging chatter
follow before the agent is asked to recall them. Nothing here calls an
LLM — it's a fixed transcript so the demo runs identically every time,
with zero API keys required.
"""

SYSTEM_PROMPT = (
    "You are a senior backend engineering assistant helping debug and ship "
    "a production incident fix. Be precise about identifiers, file paths, "
    "and ticket numbers — engineers will act on exactly what you say."
)

# The facts the demo will ask about at the very end, after eviction pressure.
PLANTED_FACTS = {
    "connection_id": "db_prod_7743a1",
    "ticket_id": "INC-88214",
    "file_path": "src/services/payment_gateway.py",
    "commit_hash": "a1b2c3d9",
}

CONVERSATION: list[tuple[str, str]] = [
    ("user", "Hey, we've got a production incident. Ticket is INC-88214."),
    ("assistant", "Got it — I've logged INC-88214 as the active incident. What are we seeing?"),
    ("user", "Payments are silently failing. The DB connection ID in the pool is db_prod_7743a1, "
             "can you check if it's saturated?"),
    ("assistant", "Noted: production connection db_prod_7743a1. Checking pool saturation now."),
    ("tool", "connection_pool_status(db_prod_7743a1) -> {active: 98, max: 100, waiting: 42, "
             "avg_wait_ms: 4300, errors_last_5m: 217}"),
    ("assistant", "That pool is nearly maxed out with a large wait queue. This lines up with the "
                  "silent payment failures — requests are timing out waiting for a connection."),
    ("user", "OK. The relevant code is in src/services/payment_gateway.py. Can you look at it?"),
    ("assistant", "Reading src/services/payment_gateway.py now."),
    ("tool", "read_file(src/services/payment_gateway.py) -> "
             "'class PaymentGateway:\\n    def __init__(self, pool):\\n        self.pool = pool\\n\\n"
             "    def charge(self, order_id, amount):\\n        with self.pool.acquire() as conn:\\n"
             "            return conn.execute(CHARGE_SQL, order_id, amount)\\n'"
             " (truncated, 3400 more chars)"),
    ("assistant", "The gateway acquires a connection per charge() call and never sets a timeout, "
                  "so requests pile up indefinitely under load instead of failing fast."),
    ("user", "Makes sense. What's the fix?"),
    ("assistant", "Add an acquire timeout and a bounded retry with backoff, so we fail fast instead "
                  "of exhausting the pool."),
    ("user", "Do it. Also check if there were recent deploys around when this started."),
    ("assistant", "Checking recent deploy history."),
    ("tool", "git_log(src/services/payment_gateway.py, limit=5) -> "
             "'a1b2c3d9 refactor: pool acquisition in PaymentGateway (2 days ago)\\n"
             "9f0e1d2c chore: bump httpx (5 days ago)\\n"
             "77aa88bb fix: retry on 502 from PSP (9 days ago)\\n'"),
    ("assistant", "Commit a1b2c3d9, two days ago, refactored pool acquisition in this exact file. "
                  "Strong candidate for the regression."),
    ("user", "Great catch. Show me the diff for a1b2c3d9."),
    ("tool", "git_show(a1b2c3d9) -> '--- a/src/services/payment_gateway.py\\n"
             "+++ b/src/services/payment_gateway.py\\n"
             "@@ -12,7 +12,7 @@\\n-        with self.pool.acquire(timeout=2.0) as conn:\\n"
             "+        with self.pool.acquire() as conn:\\n' (diff truncated)"),
    ("assistant", "Confirmed — commit a1b2c3d9 removed the timeout=2.0 argument from "
                  "pool.acquire(). That's the regression."),
    ("user", "Revert just that line and run the unit tests for payment_gateway."),
    ("assistant", "Restoring timeout=2.0 in src/services/payment_gateway.py and running tests."),
    ("tool", "run_tests(tests/test_payment_gateway.py) -> "
             "'12 passed, 0 failed in 1.84s'"),
    ("assistant", "All 12 tests pass with the timeout restored."),
    ("user", "Good. While that's running, pull up error rates for the last hour on staging too, "
             "just so we have context before we ship to prod."),
    ("tool", "metrics_query(env=staging, metric=error_rate, window=1h) -> "
             "'[0.4%, 0.3%, 0.5%, 0.4%, 0.6%, 0.3%]'"),
    ("assistant", "Staging error rate is stable around 0.3-0.6%, nothing unusual there."),
    ("user", "Fine, staging isn't the problem. Let's talk about the retry/backoff piece — what "
             "policy do you suggest?"),
    ("assistant", "Exponential backoff starting at 50ms, capped at 800ms, with 3 attempts. That "
                  "keeps latency bounded while giving the pool a chance to free up under transient load."),
    ("user", "Sounds right. Implement that in the same file."),
    ("assistant", "Adding the retry/backoff wrapper around conn.execute in "
                  "src/services/payment_gateway.py now."),
    ("tool", "apply_patch(src/services/payment_gateway.py) -> 'patch applied, 14 lines changed'"),
    ("user", "Run the full test suite, not just payment_gateway, to make sure nothing else broke."),
    ("tool", "run_tests(tests/) -> '311 passed, 0 failed in 42.6s'"),
    ("assistant", "Full suite is green — 311 passed."),
    ("user", "Nice. Let's also double check the on-call rotation, I don't want to page the wrong person."),
    ("tool", "oncall_lookup(team=payments) -> "
             "'{primary: \"j.alvarez\", secondary: \"r.chen\", escalation_after_min: 15}'"),
    ("assistant", "Primary on-call is j.alvarez, secondary r.chen, with escalation after 15 minutes."),
    ("user", "Perfect. Draft the deploy summary for the release channel."),
    ("assistant", "Draft: 'Fix: restore pool.acquire() timeout in PaymentGateway (regression from a "
                  "recent refactor) and add bounded retry/backoff. Full suite green, 311 passed.'"),
    ("user", "One more thing before you go — quick recap. What was the production DB connection ID "
             "again, and which ticket was this all under?"),
]
