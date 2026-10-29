# Development Costs

## 1. Purpose and Scope

This document tracks AWS costs associated with developing and validating the Bedrock RAG Lab. It is an extensible technical record for model inference, token consumption, embedding generation, storage, vector search, request-based charges, provisioned resources, and other AWS services introduced by future units.

Resources without token-based pricing are not assumed to be free. A cost is recorded only when supporting usage and pricing evidence exists. If either is unavailable, the cost is reported as unknown or not yet estimated rather than as zero.

## 2. Cost Estimation Methodology

For token-based inference, this project uses:

```text
estimated_cost =
    (input_tokens × input_rate / 1,000,000)
  + (output_tokens × output_rate / 1,000,000)
```

Cost records distinguish:

- **Offline estimate:** projected consumption before an invocation.
- **Observed usage:** usage reported by the service.
- **Reference pricing:** model-specific pricing assumptions used in a calculation.
- **Estimated cost:** a calculation from observed or estimated usage and reference pricing.
- **Verified billed cost:** an amount independently confirmed through billing records.
- **Credits or educational-account adjustments:** account-specific offsets recorded only when independently verified.

Calculations retain sufficient decimal precision to avoid hiding small costs through rounding. Each pricing reference should identify its source, applicable model, region or inference profile, billing modality, and verification date when available. A price that cannot be independently verified is labeled as a reference assumption rather than an official verified rate.

No cost in this document has been verified against an AWS invoice, AWS Billing, or AWS Cost Explorer.

## 3. Cumulative Cost Summary

| Experiment | Invocations | Estimated subtotal |
|---|---:|---:|
| Stage 1B — Unit 05: Nova Lite | 2 | USD 0.00018096 |
| Stage 1B — Unit 05: Claude Haiku 4.5 | 1 | USD 0.00198330 |
| **Total recorded** | **3** | **USD 0.00216426** |

The **cumulative estimated cost of recorded inference experiments** is USD 0.00216426:

```text
USD 0.00018096 + USD 0.00198330 = USD 0.00216426
```

This is not the complete AWS bill for the project. Other AWS resources, credits, taxes, and account-specific billing conditions may not be represented.

## 4. Stage 1B — Unit 05

### 4.1 Amazon Nova Lite

#### Configuration

| Setting | Value |
|---|---|
| Model | `amazon.nova-lite-v1:0` |
| Region | `us-east-1` |
| Inference modality | On-demand |
| Builder | `build_converse_request()` |
| Temperature | `0` |
| Maximum output tokens | `1024` |
| Live invocations | `2` |

#### Reference pricing

| Usage | Reference price |
|---|---:|
| Input | USD 0.06 / 1M tokens |
| Output | USD 0.24 / 1M tokens |

These are the reference rates used for the original laboratory calculations. They are preserved as historical pricing assumptions; the exact rate rows could not be independently confirmed from the dynamically rendered AWS pricing page during the documentation review on 2026-10-29.

#### Offline input measurements

| Measurement | Value |
|---|---:|
| Fixture UTF-8 size | 1,357 bytes |
| Fixture Unicode characters | 1,351 |
| Fixture whitespace-delimited words | 204 |
| System prompt characters | 391 |
| System prompt words | 60 |
| Combined content characters | 1,742 |
| Compact serialized request characters | 1,921 |

These measurements are not authoritative token counts. The initial input estimate was 450–550 tokens.

#### Initial cost estimate

Using 500 input tokens and 300 output tokens:

```text
Input:   500 × 0.06 / 1,000,000 = USD 0.000030
Output:  300 × 0.24 / 1,000,000 = USD 0.000072

Estimated total:               USD 0.000102
```

#### Observed usage and estimated cost

Both invocations reported:

```text
input_tokens=348
output_tokens=290
total_tokens=638
```

Estimated cost per invocation:

```text
Input:   348 × 0.06 / 1,000,000 = USD 0.00002088
Output:  290 × 0.24 / 1,000,000 = USD 0.00006960

Estimated cost per invocation: USD 0.00009048
```

For two invocations:

```text
2 × USD 0.00009048 = USD 0.00018096
```

**Nova Lite estimated subtotal: USD 0.00018096.**

#### Conservative configured-limit scenario

Using 550 estimated input tokens and the full 1,024-token output allowance:

```text
Input:    550 × 0.06 / 1,000,000 = USD 0.00003300
Output: 1,024 × 0.24 / 1,000,000 = USD 0.00024576

Estimated total:                 USD 0.00027876
```

`maxTokens` is an output limit, not guaranteed billed consumption. This scenario is not a universal maximum cost because input size and applicable pricing may vary.

#### Execution outcomes

Both invocations reached the Bedrock Runtime and returned complete responses, but neither produced a validated `FAQExtraction`.

The second invocation classified its parsing failure as `failure_detail=invalid_json`. The first invocation predates the more precise failure classification, so no unsupported failure cause is assigned retroactively.

### 4.2 Claude Haiku 4.5 Structured Outputs

#### Configuration

| Setting | Value |
|---|---|
| Model | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| Inference profile | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Source region | `us-east-1` |
| Inference modality | Standard, geographic cross-region, on-demand |
| Builder | `build_structured_converse_request()` |
| Temperature | `0` |
| Maximum output tokens | `1024` |
| Live invocations | `1` |

#### Reference pricing

| Usage | Global reference | US geographic reference used |
|---|---:|---:|
| Input / 1M tokens | USD 1.00 | USD 1.10 |
| Output / 1M tokens | USD 5.00 | USD 5.50 |

These are the historical pricing assumptions used for the selected `us.` inference profile and its laboratory calculation. Anthropic's pricing documentation, accessed on 2026-10-29, lists USD 1.00 input and USD 5.00 output per million tokens for Claude Haiku 4.5 and describes a 10% premium for regional endpoints. That source is supplementary model-provider information, not the authoritative source for an AWS bill. The exact AWS Bedrock rate row could not be independently captured from the dynamically rendered AWS pricing page, so the USD 1.10 and USD 5.50 figures remain reference assumptions requiring confirmation rather than verified AWS rates.

#### Observed usage and estimated cost

The successful live invocation reported:

```text
invocation_attempted=true
converse_response_received=true
stop_reason=end_turn
input_tokens=593
output_tokens=242
total_tokens=835
response_extraction_succeeded=true
schema_validation_succeeded=true
faq_entries_extracted=2
```

Using the recorded US geographic reference prices:

```text
Input:   593 × 1.10 / 1,000,000 = USD 0.00065230
Output:  242 × 5.50 / 1,000,000 = USD 0.00133100

Estimated total:                USD 0.00198330
```

**Claude Haiku 4.5 estimated subtotal: USD 0.00198330.**

Estimated cost per validated FAQ record:

```text
USD 0.00198330 / 2 = USD 0.00099165
```

This is a descriptive measurement from one successful invocation, not a statistically representative production cost.

#### Conservative configured-limit scenario

Using 593 observed input tokens and the full output limit of 1,024 tokens:

```text
Input:    593 × 1.10 / 1,000,000 = USD 0.00065230
Output: 1,024 × 5.50 / 1,000,000 = USD 0.00563200

Estimated total:                 USD 0.00628430
```

This scenario applies only to the observed input size and assumed rates. It is not a universal financial ceiling.

#### Execution outcome

The invocation completed successfully. The response passed the existing parser and Pydantic validation and produced two FAQ records. Structural validation does not establish verified semantic fidelity.

### 4.3 Experimental Comparison

| Metric | Nova Lite | Claude Haiku 4.5 |
|---|---:|---:|
| Live invocations | 2 | 1 |
| Input tokens per invocation | 348 | 593 |
| Output tokens per invocation | 290 | 242 |
| Total tokens per invocation | 638 | 835 |
| Estimated cost per invocation | USD 0.00009048 | USD 0.00198330 |
| Estimated experiment subtotal | USD 0.00018096 | USD 0.00198330 |
| Validated FAQ extraction | No | Yes |
| Validated FAQ records | 0 | 2 |

The Claude invocation had an estimated per-invocation cost approximately 21.9 times the Nova Lite invocation. This is not evidence that Claude is 21.9 times less efficient. The experiments differ in model, tokenization, prompt construction, output constraints, and execution outcomes.

The sample is insufficient to establish production reliability, general cost efficiency, or statistical success rates. No comparable cost per successful extraction can be inferred for Nova Lite because no successful Nova Lite extraction was observed.

## 5. Cost Tracking Convention for Future Units

Each future unit that incurs AWS costs should record:

1. Stage and unit identifier.
2. Experiment purpose.
3. AWS service and resource or model.
4. Region and inference profile, when applicable.
5. Billing modality.
6. Number of operations or invocations.
7. Observed usage metrics.
8. Pricing assumptions and sources.
9. Estimated cost calculation.
10. Execution outcome.
11. Billing limitations.
12. Contribution to the cumulative summary.

For non-token-based services, use the applicable billing unit instead of forcing the inference formula. Examples include embedding input tokens, storage in GB-months, vector operations, request counts, provisioned capacity, and data transfer.

Only supported figures should be documented. If an operation has no verified usage or price, record its cost as **unknown or not yet estimated**, not zero. Units that have not generated cost evidence should not receive empty or speculative cost sections.

## 6. Pricing References

The following references were reviewed on 2026-10-29:

- [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) — authoritative AWS billing reference. The page was accessible, but its dynamically rendered model-rate rows did not permit independent confirmation of the exact historical rates used above in the current review environment.
- [Amazon Bedrock model card: Anthropic Claude Haiku 4.5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html) — AWS documentation for availability, capabilities, and billing context. It directs pricing questions to the Bedrock pricing page and explains that third-party model charges appear through AWS Marketplace.
- [Anthropic Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing) — supplementary provider information supporting the global Claude Haiku 4.5 reference and regional premium. It is not an authoritative AWS billing record.

Reference pricing can change. Future entries should record the pricing source and verification date without silently rewriting historical assumptions or calculations.

## 7. Billing Limitations

- All totals are estimates derived from observed usage and reference pricing, not verified charges.
- No AWS Billing or Cost Explorer charge was checked for these experiments.
- The Vocareum educational account may apply credits, allowances, taxes, or billing arrangements that are not visible through smoke-test responses.
- Bedrock responses report usage but do not establish the final account charge.
- The cumulative summary excludes unsupported estimates for storage, networking, logging, request charges, and other AWS resources.
- Pricing varies by model, region, inference profile, billing modality, and time; configured token limits do not guarantee actual consumption.
