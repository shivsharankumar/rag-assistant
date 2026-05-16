# ADR 002: Use AWS Bedrock Titan Text Embeddings V2

## Status
Accepted (2026-05-19)

## Context
Need an embedding model. Options:
- OpenAI text-embedding-3-large/small
- Cohere embed-v3
- AWS Bedrock Titan v2
- Self-hosted (e5-large, BGE)

## Decision
AWS Bedrock Titan Text Embeddings V2 (1024 dim, normalized).

## Rationale
1. Single-vendor consistency with our LLM (Claude on Bedrock)
2. Same AWS region as RDS, ECS, S3 → no cross-region latency or egress
3. IAM-based auth — no separate API key to rotate
4. ~$0.02 per million tokens — comparable to OpenAI text-embedding-3-small
5. Comparable benchmark scores on MTEB

## Consequences
- Locked to AWS region availability (us-east-1 has full coverage)
- If we swap models, must re-embed entire corpus (no shared vector space)
- Self-hosted would be ~50% cheaper at scale but adds operational burden

## Revisit when
- Annual embedding cost exceeds $500/month → consider self-hosting
- Benchmark shows >10% retrieval quality lift from another model