"""Bedrock client wrappers for embeddings and LLM calls."""
import json
import logging
from functools import lru_cache
import boto3
from src.rag.config import settings
from langsmith import traceable

log = logging.getLogger(__name__)
# Add at the top with other constants
HAIKU = "minimax.minimax-m2.5"
SONNET = "openai.gpt-oss-20b-1:0"


from typing import Iterator
import re

# Add this helper near the top
_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>\s*", re.DOTALL | re.IGNORECASE)

def _strip_reasoning(text: str) -> str:
    """Remove <reasoning>...</reasoning> blocks that Claude sometimes emits."""
    return _REASONING_RE.sub("", text).strip()
# def claude_stream(
#     prompt: str,
#     system: str = "",
#     model_id: str = HAIKU,
#     max_tokens: int = 1024,
#     temperature: float = 0.0,
# ) -> Iterator[str]:
#     """
#     Stream Claude response chunks via Bedrock invoke_model_with_response_stream.
#     Yields text deltas as they arrive.
#     """
#     client = get_bedrock_runtime()
#     body = json.dumps({
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "system": system,
#         "messages": [{"role": "user", "content": prompt}],
#     })
    
#     response = client.invoke_model_with_response_stream(
#         modelId=model_id, body=body,
#         contentType="application/json", accept="application/json",
#     )
    
#     for event in response["body"]:
#         chunk = json.loads(event["chunk"]["bytes"])
#         if chunk.get("type") == "content_block_delta":
#             delta = chunk.get("delta", {})
#             if delta.get("type") == "text_delta":
#                 yield delta.get("text", "")

def claude_stream(prompt, system="", model_id=HAIKU, max_tokens=1024, temperature=0.0):
    client = get_bedrock_runtime()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model_with_response_stream(
        modelId=model_id, body=body,
        contentType="application/json", accept="application/json",
    )
    
    in_reasoning = False
    buffer = ""
    
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk.get("type") == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                buffer += text
                
                # State machine: skip everything inside <reasoning>...</reasoning>
                output = ""
                while buffer:
                    if not in_reasoning:
                        idx = buffer.find("<reasoning>")
                        if idx == -1:
                            # Might be mid-tag, hold back last 11 chars
                            if len(buffer) > 11:
                                output += buffer[:-11]
                                buffer = buffer[-11:]
                            break
                        else:
                            output += buffer[:idx]
                            buffer = buffer[idx + len("<reasoning>"):]
                            in_reasoning = True
                    else:
                        end_idx = buffer.find("</reasoning>")
                        if end_idx == -1:
                            buffer = ""  # discard, waiting for end tag
                            break
                        else:
                            buffer = buffer[end_idx + len("</reasoning>"):]
                            in_reasoning = False
                
                if output:
                    yield output
    
    # Flush remaining buffer
    if buffer and not in_reasoning:
        yield buffer

# def claude_invoke(
#     prompt: str,
#     system: str = "",
#     model_id: str = HAIKU,
#     max_tokens: int = 1024,
#     temperature: float = 0.0,
# ) -> str:
#     """
#     Call Claude via Bedrock with a single user message.
    
#     Args:
#         prompt: The user message
#         system: System prompt (instructions about role/behavior)
#         model_id: HAIKU (cheap/fast) or SONNET (smart/expensive)
#         max_tokens: Cap on response length
#         temperature: 0 = deterministic, 1 = creative. RAG wants 0.
    
#     Returns:
#         The text response from Claude.
#     """
#     client = get_bedrock_runtime()
    
#     body = json.dumps({
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "system": system,
#         "messages": [{"role": "user", "content": prompt}],
#     })
    
#     response = client.invoke_model(
#         modelId=model_id,
#         body=body,
#         contentType="application/json",
#         accept="application/json",
#     )
    
#     result = json.loads(response["body"].read())
#     print("Raw Bedrock response:", result)  # Debug log to inspect the full response structure
#     # Anthropic responses have a content array with text blocks
#     return result
def claude_invoke(
    prompt: str,
    system: str = "",
    model_id: str = HAIKU,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    client = get_bedrock_runtime()

    # OpenAI-compatible format (not Anthropic format)
    body = json.dumps({
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    })

    response = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    # OpenAI-style response format
    raw_text=result["choices"][0]["message"]["content"]
    return _strip_reasoning(raw_text)

@lru_cache(maxsize=1)
def get_bedrock_runtime():
    """
    Cached Bedrock runtime client.
    
    @lru_cache ensures we create ONE client per process — boto3 clients
    are thread-safe and expensive to construct. Recreating per call
    would add ~100ms latency.
    """
    session = boto3.Session(profile_name=settings.aws_profile)
    return session.client("bedrock-runtime", region_name=settings.aws_region)


# Titan Text Embeddings V2 model ID
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024

@traceable(name="bedrock_embed", run_type="embedding")
def embed_text(text: str, dimensions: int = EMBEDDING_DIM) -> list[float]:
    """
    Embed a single piece of text using Titan v2.
    
    Args:
        text: Input text (max ~8000 tokens / ~30k chars)
        dimensions: 1024 (default), 512, or 256. Lower = cheaper but less precise.
    
    Returns:
        List of `dimensions` floats.
    """
    if not text.strip():
        raise ValueError("Cannot embed empty text")
    
    client = get_bedrock_runtime()
    
    body = json.dumps({
        "inputText": text,
        "dimensions": dimensions,
        "normalize": True,  # Returns unit vectors; faster cosine search
    })
    
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    
    result = json.loads(response["body"].read())
    return result["embedding"]

@traceable(name="bedrock_claude", run_type="llm")
def embed_batch(texts: list[str], dimensions: int = EMBEDDING_DIM) -> list[list[float]]:
    """
    Embed a list of texts. Titan v2 doesn't support native batching via Bedrock,
    so we loop (still fast since each call is ~50ms).
    
    For real production with high volume, you'd use Bedrock batch inference jobs.
    """
    return [embed_text(t, dimensions) for t in texts]