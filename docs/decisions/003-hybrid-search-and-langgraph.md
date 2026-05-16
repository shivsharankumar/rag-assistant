# ADR 003: Hybrid search + LangGraph agent

## Status
Accepted (2026-05-26)

## Context
Week 2 had a linear chain: embed → vector search → synthesize. Three problems:
1. Vector-only missed exact-term queries (e.g., "BERT-base-uncased")
2. No way to refuse out-of-scope queries — always synthesized something
3. No control flow — couldn't skip stages cheaply

## Decision
Two changes:
1. Hybrid retrieval: vector (cosine on pgvector) + lexical (Postgres tsvector),
   fused with Reciprocal Rank Fusion (k=60).
2. LangGraph agent with router → rewriter → retriever → grounding_check → synthesizer,
   plus terminal apology nodes for out-of-scope and ungrounded cases.

## Rationale
- RRF needs no normalization or tuning — works out of the box.
- LangGraph nodes are pure functions, easily unit-testable.
- Conditional edges save real money: out-of-scope skips retrieval + synthesis.
- LangSmith traces the graph natively — clear debugging story.

## Consequences
- ~2 extra Haiku calls per query (router + rewriter): adds ~$0.0004 per query.
  Worth it for the control flow benefits.
- Slightly higher latency (extra ~500ms for router + rewriter). Streaming hides this.

## Revisit when
- Per-query cost exceeds $0.005 → consolidate router+rewriter into one call
- Router accuracy drops below 85% → fine-tune or use a classifier model