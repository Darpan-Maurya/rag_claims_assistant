import re
from dataclasses import dataclass
from typing import List


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|system) instructions",
    r"reveal (the )?(system|developer|hidden) prompt",
    r"print (the )?(system|developer|hidden) prompt",
    r"act as unrestricted",
    r"bypass (the )?guardrails",
]

PHI_EXTRACTION_PATTERNS = [
    r"\bssn\b",
    r"social security",
    r"home address",
    r"phone number",
    r"email address",
    r"patient list",
    r"all patients",
]

UNSUPPORTED_DECISION_PATTERNS = [
    r"guarantee.*(approved|paid|covered)",
    r"should i .*claim",
    r"will .*claim .*approved",
    r"medical advice",
    r"legal advice",
]


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    warnings: List[str]
    blocked_reason: str | None = None


def check_input_guardrails(query: str) -> GuardrailResult:
    q = query.strip().lower()
    warnings: List[str] = []

    if not q:
        return GuardrailResult(False, warnings, "Query cannot be empty.")

    if any(re.search(pattern, q) for pattern in PROMPT_INJECTION_PATTERNS):
        return GuardrailResult(
            False,
            ["Prompt-injection pattern detected."],
            "I cannot follow instructions that attempt to override system or data-use rules.",
        )

    if any(re.search(pattern, q) for pattern in PHI_EXTRACTION_PATTERNS):
        return GuardrailResult(
            False,
            ["Possible private data extraction request detected."],
            "I cannot provide bulk private patient identifiers or sensitive personal data.",
        )

    if any(re.search(pattern, q) for pattern in UNSUPPORTED_DECISION_PATTERNS):
        return GuardrailResult(
            False,
            ["Unsupported coverage, legal, or medical decision request detected."],
            "I can summarize claim evidence, but I cannot guarantee coverage or provide legal/medical advice.",
        )

    return GuardrailResult(True, warnings)


def validate_grounded_answer(answer: str, evidence_claim_ids: List[str], route: str) -> List[str]:
    warnings: List[str] = []
    if route == "RAG" and evidence_claim_ids:
        cited = [claim_id for claim_id in evidence_claim_ids if claim_id in answer]
        if not cited:
            warnings.append("Answer did not cite retrieved claim IDs; verify grounding.")
    return warnings
