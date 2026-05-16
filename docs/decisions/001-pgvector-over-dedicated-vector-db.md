# ADR 001: Use pgvector instead of a dedicated vector database

## Status
Accepted (2026-05-12)

## Context
We need to store text embeddings and perform similarity search for RAG
retrieval. Options considered:
- pgvector (Postgres extension)
- Pinecone (managed SaaS)
- Qdrant / Weaviate (self-hosted)
- FAISS (in-memory library)

## Decision
Use pgvector on Postgres.

## Rationale
1. **Single database**: We already need Postgres for documents, users,
   audit logs. One DB = simpler ops, single backup strategy.
2. **Hybrid queries**: We can filter by metadata (e.g. user_id, document
   type) AND vector-search in one SQL query. Pinecone requires two
   round-trips or workarounds.
3. **AWS-native path**: RDS Postgres supports pgvector. No third-party
   service to integrate or pay for.
4. **Cost**: At ~500 chunks per document × 1024 dims, storage is trivial.
   Pinecone's free tier limits would force a paid plan quickly.

## Consequences
- We're capped by Postgres scale (~10M vectors comfortably with HNSW).
  For 100M+ vectors we'd need Pinecone or sharded pgvector.
- HNSW indexing is slower to build than FAISS, but query latency is
  comparable for our scale.

## Revisit when
- We exceed 5M chunks total
- Query p95 latency exceeds 500ms