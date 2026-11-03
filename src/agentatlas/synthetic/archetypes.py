"""Latent capability groups used to synthesise an enterprise agent estate.

Each archetype is a hidden "true" group. The ML layer never sees the group label;
it is used only to score clustering (ARI/NMI) and to define benchmark ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Archetype:
    key: str
    department: str
    capabilities: tuple[str, ...]
    tools: tuple[str, ...]
    categories: tuple[str, ...]
    phrases: tuple[str, ...]
    data_classification: str
    latency_ms: float  # median p95 latency
    cost_usd: float  # median cost per task
    task_phrasings: dict[str, str] = field(default_factory=dict)


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        "sql_analytics", "Analytics",
        ("sql", "data_analysis", "visualization"),
        ("snowflake_query", "postgres_query", "chart_render"),
        ("sql_generation", "metric_explanation"),
        ("writes SQL against the warehouse", "answers analytics questions",
         "builds charts from query results", "explains KPI movements"),
        "internal", 1800, 0.020,
        {"sql_generation": "Translate a business question into a warehouse SQL query and run it",
         "metric_explanation": "Explain why weekly revenue KPI dropped using warehouse data"},
    ),
    Archetype(
        "fraud_detection", "Finance",
        ("fraud_detection", "transaction_monitoring", "anomaly_detection"),
        ("txn_lookup", "risk_score_api", "case_create"),
        ("fraud_triage", "alert_review"),
        ("monitors card transactions for fraud", "scores transaction risk",
         "opens fraud investigation cases", "detects anomalous payment patterns"),
        "restricted", 900, 0.015,
        {"fraud_triage": "Triage a suspicious card payment and decide whether to open a case",
         "alert_review": "Review a batch of transaction monitoring alerts for false positives"},
    ),
    Archetype(
        "customer_support", "Support",
        ("customer_support", "ticket_routing", "summarization"),
        ("zendesk_api", "kb_search", "email_send"),
        ("ticket_resolution", "ticket_triage"),
        ("resolves customer tickets", "routes support requests to queues",
         "searches the knowledge base", "drafts customer replies"),
        "confidential", 2500, 0.010,
        {"ticket_resolution": "Resolve a customer complaint about a failed refund",
         "ticket_triage": "Route incoming helpdesk requests to the right support queue"},
    ),
    Archetype(
        "code_generation", "Engineering",
        ("code_generation", "code_review", "testing"),
        ("github_api", "ci_trigger", "sandbox_exec"),
        ("code_fix", "code_review"),
        ("writes and fixes Python code", "reviews pull requests",
         "generates unit tests", "runs code in a sandbox"),
        "internal", 6000, 0.060,
        {"code_fix": "Fix a failing unit test in a Python service and open a pull request",
         "code_review": "Review a pull request diff for bugs and style problems"},
    ),
    Archetype(
        "document_extraction", "Operations",
        ("document_extraction", "ocr", "classification"),
        ("ocr_api", "s3_read", "schema_validate"),
        ("invoice_extraction", "contract_parsing"),
        ("extracts fields from scanned invoices", "runs OCR on PDFs",
         "classifies inbound documents", "validates extracted data against schemas"),
        "confidential", 4000, 0.030,
        {"invoice_extraction": "Pull vendor, amount and due date out of scanned supplier invoices",
         "contract_parsing": "Parse key clauses and dates from uploaded agreement PDFs"},
    ),
    Archetype(
        "hr_policy", "HR",
        ("hr_policy_qa", "rag", "summarization"),
        ("workday_api", "policy_search"),
        ("policy_question", "benefits_question"),
        ("answers employee HR policy questions", "explains benefits enrollment",
         "retrieves handbook passages", "summarises leave policies"),
        "confidential", 2000, 0.008,
        {"policy_question": "Answer an employee question about the parental leave policy",
         "benefits_question": "Explain which health benefit plan options an employee can enrol in"},
    ),
    Archetype(
        "research_rag", "Strategy",
        ("rag", "web_research", "summarization", "citation"),
        ("vector_search", "web_search", "pdf_read"),
        ("research_brief", "literature_review"),
        ("researches topics on the web", "writes cited research briefs",
         "retrieves internal reports", "reviews academic literature"),
        "internal", 9000, 0.080,
        {"research_brief": "Produce a cited market brief on competitors in the EU payments space",
         "literature_review": "Summarise recent papers on retrieval augmented generation evaluation"},
    ),
    Archetype(
        "finance_reconciliation", "Finance",
        ("reconciliation", "accounting", "sql"),
        ("erp_query", "ledger_api", "spreadsheet_write"),
        ("gl_reconciliation", "variance_analysis"),
        ("reconciles general ledger accounts", "matches ERP entries to bank statements",
         "explains budget variances", "prepares month end close workpapers"),
        "confidential", 3000, 0.025,
        {"gl_reconciliation": "Reconcile the cash general ledger account against bank statements",
         "variance_analysis": "Explain the budget versus actual variance for Q3 opex"},
    ),
    Archetype(
        "security_triage", "Security",
        ("security_triage", "log_analysis", "threat_intel"),
        ("siem_query", "edr_isolate", "ticket_create"),
        ("alert_triage", "incident_summary"),
        ("triages SIEM security alerts", "enriches indicators with threat intel",
         "isolates compromised endpoints", "analyses authentication logs"),
        "restricted", 1500, 0.020,
        {"alert_triage": "Triage a SIEM alert for impossible travel logins",
         "incident_summary": "Summarise a phishing incident timeline for the security team"},
    ),
    Archetype(
        "marketing_content", "Marketing",
        ("content_generation", "seo", "brand_voice"),
        ("cms_publish", "image_gen", "analytics_api"),
        ("campaign_copy", "seo_article"),
        ("writes marketing campaign copy", "optimises articles for SEO",
         "keeps content on brand voice", "publishes posts to the CMS"),
        "public", 5000, 0.040,
        {"campaign_copy": "Write email campaign copy for a product launch in our brand voice",
         "seo_article": "Draft a search optimised blog article about expense automation"},
    ),
    Archetype(
        "it_ops", "IT",
        ("incident_response", "log_analysis", "automation"),
        ("k8s_api", "pagerduty_api", "runbook_exec"),
        ("incident_remediation", "change_review"),
        ("responds to production incidents", "executes remediation runbooks",
         "restarts failing kubernetes workloads", "reviews change requests"),
        "internal", 2200, 0.020,
        {"incident_remediation": "Remediate a crashlooping kubernetes deployment paging on-call",
         "change_review": "Assess the risk of a proposed production database change"},
    ),
    Archetype(
        "legal_review", "Legal",
        ("contract_review", "compliance", "summarization"),
        ("clm_api", "policy_search", "redline"),
        ("contract_review", "compliance_check"),
        ("reviews commercial contracts", "redlines non standard clauses",
         "checks regulatory compliance", "summarises legal obligations"),
        "confidential", 7000, 0.070,
        {"contract_review": "Redline a vendor master services agreement for risky indemnity terms",
         "compliance_check": "Check a new data retention process against GDPR obligations"},
    ),
)

ARCHETYPE_BY_KEY = {a.key: a for a in ARCHETYPES}

# A deliberately rare capability: a handful of agents in a 10k estate have it.
NEEDLE_CAPABILITY = "sanctions_screening"
NEEDLE_CATEGORY = "sanctions_screening"
NEEDLE_PHRASES = ("screens counterparties against OFAC sanctions lists",
                  "flags politically exposed persons")
NEEDLE_TASK = "Screen a new supplier and its directors against international sanctions lists"

GENERIC_PHRASES = (
    "uses an LLM with tool calling", "built for internal teams", "handles multi step tasks",
    "integrates with enterprise systems", "supports human in the loop review",
    "logs actions for audit",
)

FRAMEWORKS = ("langgraph", "crewai", "autogen", "openai_agents_sdk", "bedrock_agents",
              "semantic_kernel", "custom")
MODELS = ("claude-sonnet", "claude-haiku", "gpt-4o", "gpt-4o-mini", "llama-3-70b", "mistral-large")
CLASSIFICATION_ORDER = ("public", "internal", "confidential", "restricted")


def classification_rank(level: str) -> int:
    return CLASSIFICATION_ORDER.index(level)
