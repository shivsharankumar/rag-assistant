# ADR 004: Reranker, self-critique, and RAGAS-style eval

## Status
Accepted (2026-06-02)

## Context
Week 3 had agent + hybrid search, but quality was unmeasured. Specific gaps:
- No precision filter — hybrid sometimes included tangential chunks
- No hallucination verification after synthesis
- No quantitative quality metrics to gate changes

## Decision
Three additions:
1. LLM-as-reranker (Haiku, listwise) between retriever and grounding
2. Critic node (Sonnet, opt-in) after synthesizer
3. Custom RAGAS-style metrics: faithfulness, relevance, context precision

## Rationale
- Listwise reranker: 1 API call vs 20, cheaper and more consistent
- Sonnet for critic (not Haiku): avoid same model self-bias
- Custom metrics over ragas library: deeper understanding, Bedrock-native, 
  no extra deps; ragas can be added later if needed
- Opt-in critic: doubles cost; user/business decides per query

## Measured impact
| Metric | Without reranker | With reranker | Lift |
|---|---|---|---|
| Faithfulness | 0.82 | 0.91 | +11% |
| Answer Relevance | 0.88 | 0.94 | +7% |
| Context Precision | 0.64 | 0.88 | +38% |

(Numbers from eval/results-week4.txt)

## Consequences
- Per-query cost increased ~60% (extra Haiku rerank call)
- Per-query cost with critic: ~5x baseline (Sonnet judge)
- Latency: +500ms rerank, +1.5s with critic
- Eval runs cost ~$0.40 each — run only on pipeline changes

## Revisit when
- Critic recall drops below 80% on labeled hallucination set
- Cohere Rerank v3 prices justify the swap (benchmark first)
- More than 30 distinct eval examples — consider RAGAS library