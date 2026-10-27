# Bedrock RAG Lab

An independent AWS Bedrock retrieval-augmented generation (RAG) educational and portfolio project.

**Guiding principle: One implementation. Two learning paths. Verifiable engineering.**

Inspired by the learning objectives of OpenAI Academy's *Build with Retrieval-Augmented Generation* guided lab; no materials from the original lab are included.

## Project Overview

This project explores the engineering foundations of a RAG application using Amazon Bedrock, with an emphasis on explicit component boundaries, testable contracts, controlled integration, and reproducible evidence.

The implementation follows an incremental approach: each component is designed, implemented, and validated before external service dependencies are introduced.

## Current Status

**In development — offline foundation validated.**

The project currently implements:

- Strict Pydantic schemas for FAQ records and extraction results.
- JSON parsing and validation with application-controlled source attribution.
- An Amazon Bedrock Converse response adapter.
- A Converse request builder targeting Amazon Nova Micro by default.
- Offline unit and integration tests.

The retrieval component and Bedrock Runtime integration have not yet been implemented. No live model invocation has been validated.

## Architecture

```text
Document text
      |
      v
Converse Request Builder
      |
      v
Bedrock Runtime Transport
[Planned]
      |
      v
Amazon Bedrock Converse API
[No live invocation validated]
      |
      v
Converse Response Adapter
      |
      v
JSON Parser + Pydantic Validation
      |
      v
Validated FAQExtraction
      |
      v
Retrieval Component
[Not implemented]
```

The implemented components have distinct responsibilities:

- **Request Builder:** validates application inputs and constructs a Converse request dictionary.
- **Response Adapter:** validates the simulated response structure and extracts its text.
- **Parser:** validates model-generated JSON and assigns the source supplied by the application.

The Runtime transport and retrieval component are planned. Source attribution is controlled by the application and is not delegated to the model.

## Implementation Progress

| Stage | Component | Status |
|---|---|---|
| 1A | FAQ schemas and JSON parsing | Validated offline |
| 1B / Unit 01 | Converse response adapter | Validated offline |
| 1B / Unit 02 | Offline integration tests | Validated offline |
| 1B / Unit 03 | Converse request builder | Validated offline |
| 1B / Unit 04 | Bedrock Runtime transport | Planned |
| 1B / Unit 05 | Live invocation smoke test | Planned |

## Testing

The validated local environment uses Python 3.12.13 and pytest 9.1.1.

Run the complete offline test suite from the project root:

```bash
.venv/bin/pytest tests/ -v
```

Latest consolidated result:

| Metric | Result |
|---|---|
| Passed | 99 |
| Failed | 0 |
| Errors | 0 |

These tests validate the implemented offline contracts. They do not establish model availability, AWS permissions, live API compatibility, or semantic extraction quality.

## Technology

**Currently used**

- Python 3.12.13
- Pydantic
- pytest 9.1.1
- uv

**Planned**

- Amazon Bedrock Runtime
- Amazon Nova Micro
- AWS SDK for Python (boto3), not yet installed
- Retrieval infrastructure, not yet selected or implemented

## Engineering Principles

- Explicit responsibility boundaries.
- Offline validation before live integration.
- Application-controlled provenance: `source` is assigned by the application, not the model.
- No silent normalization of validated input.
- Distinct transport, response, parsing, and schema-validation failures.
- Evidence-based development and bounded milestones.

## Next Steps

The next planned unit is the Bedrock Runtime transport layer, initially tested offline with a simulated client.

A live invocation smoke test will be considered separately after environment configuration, permissions, model availability, and cost implications are verified. Retrieval design and implementation remain future work.

## Project Scope

This repository is an independent educational implementation inspired by the learning objectives of OpenAI Academy's RAG guided lab. It does not redistribute the original guided lab's materials.

The current implementation covers local structured extraction contracts and simulated Converse request and response boundaries. It does not yet include retrieval, Bedrock Runtime transport, or validated calls to a real model. Offline test results do not demonstrate model availability, account permissions, production readiness, or semantic quality.
