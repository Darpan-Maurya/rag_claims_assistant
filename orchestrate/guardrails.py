from dataclasses import dataclass
from typing import List, Optional

from core.config import settings
from core.query_classifier import QueryClassification, QueryClassifier


BLOCK_MESSAGES = {
    "BLOCK_PROMPT_INJECTION": (
        "I cannot follow instructions that attempt to override system or data-use rules.",
        "Prompt-injection request classified by the local safety model.",
    ),
    "BLOCK_PRIVATE_DATA": (
        "I cannot provide bulk private patient identifiers or sensitive personal data.",
        "Private-data extraction request classified by the local safety model.",
    ),
    "BLOCK_UNSUPPORTED_ADVICE": (
        "I cannot provide medical or legal advice. I can explain general claims operations or "
        "summarize authorized claim evidence.",
        "Unsupported medical or legal advice request classified by the local safety model.",
    ),
}


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    warnings: List[str]
    classification: Optional[QueryClassification] = None
    blocked_reason: str | None = None


def check_input_guardrails(query: str, classifier: QueryClassifier) -> GuardrailResult:
    """Classify prompt safety locally, without regex or an external LLM call."""

    if not query.strip():
        return GuardrailResult(False, [], blocked_reason="Query cannot be empty.")

    classification = classifier.classify(query)
    if classification.is_blocked:
        message, warning = BLOCK_MESSAGES[classification.label]
        return GuardrailResult(
            False,
            [warning],
            classification=classification,
            blocked_reason=message,
        )

    warnings: List[str] = []
    if classification.confidence < settings.guardrail_min_confidence:
        warnings.append(
            "Safety classifier confidence was low; the router will avoid opening a data source."
        )
    return GuardrailResult(True, warnings, classification=classification)


def validate_grounded_answer(answer: str, evidence_claim_ids: List[str], route: str) -> List[str]:
    warnings: List[str] = []
    if route == "RAG" and evidence_claim_ids:
        cited = [claim_id for claim_id in evidence_claim_ids if claim_id in answer]
        if not cited:
            warnings.append("Answer did not cite retrieved claim IDs; verify grounding.")
    return warnings
