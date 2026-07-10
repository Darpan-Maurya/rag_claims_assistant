# Claims Dataset Strategy

This project uses synthetic claims by default so the app can be demonstrated,
tested, and deployed without protected health information.

## Default Dataset

- Source: `generate_mock_data.py`
- Size: 5,000 claims by default
- PHI status: synthetic, no real patient identifiers
- Purpose: production-style RAG demonstration with realistic claim metadata,
  denial patterns, payments, plan details, and service dates

The generated data is acceptable for demonstrating retrieval architecture,
guardrails, analytics routing, feedback capture, and evaluation. It should not
be used to infer real payer behavior.

## Public Synthetic Options

- CMS SynPUF / DE-SynPUF: useful for Medicare-style claims schemas and
  large-scale synthetic claims layouts.
- Synthea: useful for longitudinal synthetic patients, encounters, diagnoses,
  and claims-like exports.
- Kaggle datasets: optional only after confirming license, provenance, schema
  quality, and absence of sensitive data.

## Real-Data Readiness

The application treats future real claims as PHI-like data:

- avoid logging full queries by default
- redact patient and claim identifiers in logs
- validate schemas before indexing
- keep audit and feedback events
- never commit real claims, secrets, or exported PHI to the repository
