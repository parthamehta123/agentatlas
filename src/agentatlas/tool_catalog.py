"""Tool risk taxonomy (in production: sourced from the tool/MCP catalog or IAM scopes)."""

WRITE_TOOLS = frozenset({
    "case_create", "email_send", "ci_trigger", "sandbox_exec", "spreadsheet_write",
    "edr_isolate", "ticket_create", "cms_publish", "runbook_exec", "k8s_api", "redline",
    "shell_exec", "db_admin"})
PII_TOOLS = frozenset({"zendesk_api", "workday_api", "txn_lookup", "email_send", "db_admin"})
EXTERNAL_TOOLS = frozenset({"web_search", "email_send", "cms_publish", "image_gen",
                            "http_fetch_any"})
BLOCKED_TOOLS = frozenset({"shell_exec", "http_fetch_any", "db_admin"})
