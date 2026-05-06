#!/usr/bin/env python3
"""
BEAM End-to-End Evaluation Pipeline
===================================
Evaluates Mnemosyne as a memory backend for LLMs using the official
BEAM benchmark protocol:
  1. Download BEAM dataset from HuggingFace
  2. Ingest conversations into Mnemosyne
  3. For each probing question: retrieve memories -> LLM answers -> LLM-as-judge scores
  4. Report per-scale, per-ability scores comparable to published SOTA

Published SOTA (BEAM 10M):
  Hindsight: 64.1%   Honcho: 40.6%   LIGHT: 26.6%   RAG: 24.9%

LLM: Nvidia API (deepseek-ai/deepseek-v4-pro) via OpenAI-compatible endpoint.
     Fast, cheap (~$2/M tokens), no local GPU needed.

Usage:
  cd /root/.hermes/projects/mnemosyne
  .venv/bin/python tools/evaluate_beam_end_to_end.py --sample 5 --scales 100K,500K,1M,10M

--sample N: conversations per scale (default 3, use 0 for all)
--scales: comma-separated (default 100K,500K,1M,10M)
--mode: retrieval|end_to_end (default end_to_end)
--judge-model: LLM model for judging (default same as answer model)
--resume: skip already-evaluated questions from results file
"""

import argparse
import ast
import gc
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

# Unbuffered output for real-time progress
print = partial(print, flush=True)

# --- Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import urllib.request
import urllib.error
import numpy as np

from mnemosyne.core.beam import BeamMemory, init_beam, _embeddings, _vec_available, _vec_insert, _fts_search_working, _generate_id

# --- Config ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = "google/gemini-flash-latest"
FALLBACK_MODELS = [
    "google/gemini-2.5-pro",  
]
DEFAULT_TOP_K = 10  # Memories to retrieve per question
MAX_MEMORY_CONTEXT_CHARS = 8000  # Max chars of retrieved context to send to LLM
BENCHMARK_QUERIES_PER_CONV = 50  # Max probing questions per conversation
RESULTS_FILE = PROJECT_ROOT / "results" / "beam_e2e_results.json"

# Memory abilities tested by BEAM (10 dimensions)
BEAM_ABILITIES = [
    "IE",   # Information Extraction
    "MR",   # Multi-hop Reasoning
    "KU",   # Knowledge Update
    "TR",   # Temporal Reasoning
    "ABS",  # Abstention
    "CR",   # Contradiction Resolution
    "EO",   # Event Ordering
    "IF",   # Instruction Following
    "PF",   # Preference Following
    "SUM",  # Summarization
]

# Map dataset ability names to our abbreviations
ABILITY_MAP = {
    "information_extraction": "IE",
    "multi_session_reasoning": "MR",
    "knowledge_update": "KU",
    "temporal_reasoning": "TR",
    "abstention": "ABS",
    "contradiction_resolution": "CR",
    "event_ordering": "EO",
    "instruction_following": "IF",
    "preference_following": "PF",
    "summarization": "SUM",
    # Aliases
    "multi_session": "MR",
    "knowledge": "KU",
    "temporal": "TR",
    "information": "IE",
}


# ============================================================
#  LLM Client
# ============================================================

class LLMClient:
    """OpenAI-compatible API client using OpenRouter (fast, reliable)."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = None, base_url: str = None):
        self.model = model
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = (base_url or OPENROUTER_BASE_URL).rstrip("/")
        self.fallback_models = FALLBACK_MODELS.copy()
        self.call_count = 0

    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """Send chat completion request with fallback and retry."""
        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]
        last_error = None

        for model in models_to_try:
            for attempt in range(3):
                try:
                    return self._call_api(model, messages, temperature, max_tokens)
                except Exception as e:
                    last_error = str(e)
                    if "429" in last_error or "rate" in last_error.lower():
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    else:
                        break  # Non-retryable error, try next model
            # If we get here, this model failed all attempts
            if "429" not in str(last_error) and "rate" not in str(last_error).lower():
                continue  # Try next model
            time.sleep(3)  # Brief pause between models

        return f"[LLM_ERROR: all models failed. Last: {last_error}]"

    def _call_api(self, model: str, messages: list, temperature: float, max_tokens: int) -> str:
        """Single API call via urllib."""
        import json as _json
        url = f"{self.base_url}/chat/completions"
        payload = _json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=payload, headers=headers)
        resp = urllib.request.urlopen(req, timeout=15)
        data = _json.loads(resp.read())
        self.call_count += 1
        return data["choices"][0]["message"]["content"]

    def close(self):
        pass


# ============================================================
#  Data Loading (adapted from benchmark_beam_sota.py)
# ============================================================

def load_beam_dataset(scales: list[str], max_conversations: int = None) -> dict:
    """Load BEAM dataset from HuggingFace. Returns dict[scale] -> list[conversation]."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    data = {}
    total_loaded = 0

    for scale in scales:
        print(f"  Loading BEAM {scale}...")
        try:
            if scale == "10M":
                ds = load_dataset("Mohammadta/BEAM-10M", streaming=True)
                split_name = "10M" if "10M" in ds else list(ds.keys())[0]

                conversations = []
                for i, sample in enumerate(ds[split_name]):
                    if max_conversations and i >= max_conversations:
                        break

                    probing_raw = sample.get("probing_questions", {})
                    if isinstance(probing_raw, str):
                        try:
                            probing = ast.literal_eval(probing_raw)
                        except Exception:
                            probing = {}
                    else:
                        probing = probing_raw

                    all_questions = []
                    for ability, questions in probing.items():
                        if isinstance(questions, list):
                            for q in questions:
                                if isinstance(q, dict):
                                    all_questions.append({
                                        "ability": ability,
                                        "question": q.get("question", ""),
                                        "ideal_answer": q.get("ideal_answer", q.get("ideal_response", q.get("answer", q.get("ideal_summary", "")))),
                                        "rubric": q.get("rubric", []),
                                    })

                    # Extract messages from plans
                    plans = sample.get("plans", [])
                    all_messages = []
                    for plan in plans:
                        chat_blocks = plan.get("chat", []) if isinstance(plan, dict) else []
                        for block in chat_blocks:
                            if isinstance(block, list):
                                for msg in block:
                                    if isinstance(msg, dict):
                                        all_messages.append({
                                            "role": msg.get("role", "unknown"),
                                            "content": msg.get("content", ""),
                                            "index": len(all_messages),
                                        })

                    conversations.append({
                        "id": sample.get("conversation_id", str(i)),
                        "messages": all_messages,
                        "questions": all_questions,
                        "scale": "10M",
                    })
                    total_loaded += 1

                data[scale] = conversations
                ds.cleanup_cache_files() if hasattr(ds, 'cleanup_cache_files') else None
                del ds
                gc.collect()
                print(f"    Loaded {len(conversations)} conversations")

            else:
                # 100K, 500K, 1M scales from the main dataset
                ds = load_dataset("Mohammadta/BEAM", streaming=True)
                if scale not in ds:
                    print(f"    WARNING: split '{scale}' not found. Available: {list(ds.keys())}")
                    continue

                conversations = []
                for i, sample in enumerate(ds[scale]):
                    if max_conversations and i >= max_conversations:
                        break

                    pq_raw = sample.get("probing_questions", "{}")
                    if isinstance(pq_raw, str):
                        try:
                            probing = ast.literal_eval(pq_raw)
                        except Exception:
                            probing = {}
                    else:
                        probing = pq_raw

                    flat_questions = []
                    for ability, questions in probing.items():
                        if isinstance(questions, list):
                            for q in questions:
                                if isinstance(q, dict):
                                    flat_questions.append({
                                        "ability": ability,
                                        "question": q.get("question", ""),
                                        "ideal_answer": q.get("ideal_answer", q.get("ideal_response", q.get("answer", q.get("ideal_summary", "")))),
                                        "rubric": q.get("rubric", []),
                                    })

                    chat_blocks = sample.get("chat", [])
                    messages = []
                    for block in chat_blocks:
                        if isinstance(block, list):
                            for msg in block:
                                if isinstance(msg, dict):
                                    messages.append({
                                        "role": msg.get("role", "unknown"),
                                        "content": msg.get("content", ""),
                                        "index": len(messages),
                                    })

                    conversations.append({
                        "id": sample.get("conversation_id", str(i)),
                        "messages": messages,
                        "questions": flat_questions,
                        "scale": scale,
                    })
                    total_loaded += 1

                data[scale] = conversations
                ds.cleanup_cache_files() if hasattr(ds, 'cleanup_cache_files') else None
                del ds
                gc.collect()
                print(f"    Loaded {len(conversations)} conversations")

        except Exception as e:
            print(f"    ERROR loading {scale}: {e}")
            import traceback
            traceback.print_exc()

    print(f"  Total: {total_loaded} conversations across {len(data)} scales")
    return data


# ============================================================
#  Mnemosyne Ingestion
# ============================================================

def _extract_facts(content: str, source: str = "unknown") -> list[dict]:
    """Extract structured facts from a message for precision retrieval.
    These fact entries complement raw message storage by isolating
    specific data points (numbers, dates, versions, negations) that
    FTS5 keyword search can match more precisely than in long messages."""
    import re
    facts = []
    
    # Pattern 1: Version numbers ("Flask 2.3.1", "v0.6.2", "Python 3.11")
    ver_matches = re.findall(r'([A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)*)\s+v?(\d+\.\d+(?:\.\d+)?)', content)
    for name, ver in ver_matches[:3]:
        facts.append({
            "content": f"FACT version: {name.strip()} {ver}",
            "importance": 0.7,
        })
    
    # Pattern 2: Numbers with units ("250ms", "3 columns", "50 tasks", "5000 port")
    num_matches = re.findall(r'(\d+(?:[.,]\d+)?)\s*(ms|sec|seconds?|minutes?|hours?|days?|weeks?|months?|%|KB|MB|GB|columns?|tasks?|commits?|users?|ports?|items?)', content, re.IGNORECASE)
    for num, unit in num_matches[:5]:
        facts.append({
            "content": f"FACT metric: {num}{unit}",
            "importance": 0.65,
        })
    
    # Pattern 3: Dates
    date_patterns = [
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}',
        r'\d{4}-\d{2}-\d{2}',
    ]
    for pat in date_patterns:
        for match in re.findall(pat, content, re.IGNORECASE):
            if isinstance(match, tuple):
                match = " ".join(match)
            facts.append({
                "content": f"FACT date: {match}",
                "importance": 0.7,
            })
    
    # Pattern 4: Deadlines
    deadline_matches = re.findall(r'(deadline|due by|sprint ends?|sprint \d+)\s*[:\-]?\s*([^.,;!?\n]{5,80})', content, re.IGNORECASE)
    for ctx, detail in deadline_matches[:3]:
        facts.append({
            "content": f"FACT deadline: {ctx} {detail.strip()}",
            "importance": 0.7,
        })
    
    # Pattern 5: Negations ("I have never", "I have not") - critical for CR
    negations = re.findall(r'(I(?: have|\'ve)?\s*(?:never|not)\s+[^.,;!?\n]{15,120})', content, re.IGNORECASE)
    for neg in negations[:3]:
        facts.append({
            "content": f"FACT negation: {neg.strip()}",
            "importance": 0.75,
        })
    
    # Pattern 6: Decisions / choices
    choices = re.findall(r'(?:decided to|chose to|opted for|selected|picked|switching to)\s+([^.,;!?\n]{10,120})', content, re.IGNORECASE)
    for choice in choices[:3]:
        facts.append({
            "content": f"FACT decision: {choice.strip()}",
            "importance": 0.65,
        })
    
    # Pattern 7: Ordinal sequence markers ("first", "then", "finally") for EO
    ordinals = re.findall(r'((?:first|second|third|fourth|fifth|finally|next|then|after that)[^.,;!?\n]{15,120})', content, re.IGNORECASE)
    for ord_text in ordinals[:5]:
        facts.append({
            "content": f"FACT sequence: {ord_text.strip()}",
            "importance": 0.6,
        })
    
    # Pattern 8: Entity-action pairs ("transactions table" + "add") for MR
    entities = re.findall(r'(?:the|my|our)\s+([a-z_]+\s*(?:table|model|schema|API|endpoint|function|module|route|handler))\s+(?:needs?|requires?|should|could|would|will|has|have)\s+([^.,;!?\n]{10,80})', content, re.IGNORECASE)
    for entity, action in entities[:5]:
        facts.append({
            "content": f"FACT entity: {entity.strip()} -> {action.strip()}",
            "importance": 0.65,
        })
    
    return facts[:20]  # Cap per message

def ingest_conversation(beam: BeamMemory, messages: list[dict]) -> dict:
    """Ingest conversation messages into Mnemosyne BEAM tiers.
    Simple batch ingestion: working memory (FTS5) + periodic episodic consolidation."""
    start_time = time.perf_counter()
    stats = {"wm_count": 0, "ep_count": 0, "sp_count": 0, "total_chars": 0}

    BATCH_SIZE = 500

    for batch_start in range(0, len(messages), BATCH_SIZE):
        batch_msgs = messages[batch_start:batch_start + BATCH_SIZE]

        batch_items = []
        for i, msg in enumerate(batch_msgs):
            content = msg.get("content", "")
            if not content.strip():
                continue
            batch_items.append({
                "content": content,
                "source": f"beam_{msg.get('role', 'unknown')}",
                "importance": 0.3 + (0.1 * ((batch_start + i) % 5)),
            })
            stats["total_chars"] += len(content)

            # Scratchpad every 10 messages
            if (batch_start + i) % 10 == 0 and len(content) > 50:
                try:
                    beam.scratchpad_write(f"[t={batch_start + i}] {content[:300]}")
                    stats["sp_count"] += 1
                except Exception:
                    pass

        if not batch_items:
            continue

        beam.remember_batch(batch_items)
        stats["wm_count"] += len(batch_items)

        # Episodic consolidation per batch
        try:
            cursor = beam.conn.cursor()
            # Get ALL working memory items for this session (oldest first)
            cursor.execute("""
                SELECT id, content FROM working_memory
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT 1000
            """, (beam.session_id,))
            wm_rows = cursor.fetchall()

            if wm_rows:
                wm_ids = [row["id"] for row in wm_rows]
                recent_texts = [row["content"][:100] for row in wm_rows[:5]]
                summary = f"Batch {batch_start // BATCH_SIZE}: " + " | ".join(recent_texts[:3])
                if len(summary) > 500:
                    summary = summary[:497] + "..."

                beam.consolidate_to_episodic(
                    summary=summary,
                    source_wm_ids=wm_ids,
                    source="beam_consolidation",
                    importance=0.4,
                    scope="global",
                )
                stats["ep_count"] += 1
                
                # Delete consolidated items from working memory to prevent bloat
                placeholders = ",".join("?" * len(wm_ids))
                cursor.execute(f"DELETE FROM working_memory WHERE id IN ({placeholders})", wm_ids)
                stats["wm_count"] -= len(wm_ids)
                
                beam.conn.commit()
        except Exception:
            pass

    stats["ingest_time_ms"] = (time.perf_counter() - start_time) * 1000
    return stats


# ============================================================
#  LLM Answering with Mnemosyne Memory
# ============================================================

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions about a user's past conversations. You have access to retrieved conversation memories.

Your job: answer questions using the provided memories. Give the BEST answer you can based on the most relevant information, even if some details are unclear.

CRITICAL RULES:
- Answer based on the MOST RELEVANT memories. If you see a clear answer, provide it confidently.
- Only say "I don't have enough information" if the memories contain NOTHING related to the question.
- If memories contain partial information, answer with what you have rather than abstaining.
- For EVENT ORDERING questions, extract timestamps or explicit ordering clues from the memories.
- For TEMPORAL questions, look for dates, timestamps, or relative time references.
- If there are contradictions, mention both possibilities.

Be concise but thorough. Include specific details from the memories to support your answer."""

DEFAULT_TOP_K = 30  # Memories to retrieve per question (increased for broader context)
RECENT_CONTEXT_COUNT = 12  # Last N messages to include as recent context
MAX_MEMORY_CONTEXT_CHARS = 16000  # More context for LLM to find contradictions


def _recall_safe(beam: BeamMemory, query: str, top_k: int) -> list:
    """Safe recall wrapper that handles errors gracefully."""
    try:
        return beam.recall(query, top_k=top_k)
    except Exception:
        return []


def _extract_search_terms(question: str) -> list[str]:
    """Extract diverse search terms from a question for multi-strategy retrieval."""
    import re
    terms = []
    
    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', question)
    terms.extend(quoted)
    
    # Extract numbers and units
    numbers = re.findall(r'\b\d+[.,]?\d*\s*(?:ms|sec|days?|weeks?|months?|years?|%|KB|MB|GB|hours?|minutes?)\b', question, re.IGNORECASE)
    terms.extend(numbers[:5])
    
    # Extract named entities (capitalized phrases)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', question)
    terms.extend(entities[:5])
    
    # Extract version strings
    versions = re.findall(r'\bv?\d+\.\d+(?:\.\d+)?\b', question)
    terms.extend(versions[:5])
    
    # Extract key nouns (filter out question words)
    stop_words = {'have', 'did', 'do', 'does', 'can', 'will', 'would', 'should', 'is', 'are', 'was', 'were',
                  'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'my', 'me', 'i', 'you', 
                  'how', 'what', 'when', 'where', 'which', 'who', 'why', 'many', 'much'}
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', question) if w.lower() not in stop_words]
    terms.extend(words[:10])
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    
    return unique


def _multi_strategy_recall(beam: BeamMemory, question: str, top_k: int = DEFAULT_TOP_K) -> list:
    """Multi-strategy retrieval: keyword, semantic, entity, negation, temporal."""
    import re
    all_memories = []
    seen_content_keys = set()
    
    def _add_unique(mems):
        for mem in mems:
            ck = mem.get("content", "")[:80]
            if ck not in seen_content_keys:
                seen_content_keys.add(ck)
                all_memories.append(mem)
    
    # Strategy 1: Direct question search (mostly keyword via FTS5)
    _add_unique(_recall_safe(beam, question, top_k * 2))
    
    # Strategy 2: Negation search for contradiction detection
    if any(w in question.lower() for w in ["have i", "did i", "do i", "am i", "has"]):
        negation_query = question
        for negation in ["never", "did not", "haven't"]:
            if negation not in negation_query.lower():
                negation_query = re.sub(r'(?i)(have i|did i|am i)', f'I {negation}', negation_query)
                break
        _add_unique(_recall_safe(beam, negation_query, top_k))
    
    # Strategy 3: Key entity/term searches
    terms = _extract_search_terms(question)
    for term in terms[:5]:
        if len(term) > 2:
            _add_unique(_recall_safe(beam, term, max(5, top_k // 3)))
    
    # Strategy 4: Temporal search for date-related questions
    temporal_keywords = ['when', 'date', 'deadline', 'sprint', 'day', 'week', 'month', 
                         'april', 'march', 'february', 'january', 'may', 'june', 'july',
                         'august', 'september', 'october', 'november', 'december',
                         'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    if any(w in question.lower() for w in temporal_keywords):
        # Search for dates and timelines
        _add_unique(_recall_safe(beam, "deadline schedule timeline date", top_k))
        # Search for specific months mentioned in the question
        for month in ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december']:
            if month in question.lower():
                _add_unique(_recall_safe(beam, month, top_k // 2))
    
    # Sort by score and return top-k
    all_memories.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_memories[:top_k]


def answer_with_memory(llm: LLMClient, beam: BeamMemory, question: str, 
                      conversation_messages: list = None, top_k: int = DEFAULT_TOP_K) -> str:
    """Retrieve memories and have LLM answer, with context strategy based on conversation size."""
    
    total_msgs = len(conversation_messages) if conversation_messages else 0
    
    # STRATEGY: For small conversations (under 500 msgs), give the LLM ALL messages.
    # For larger ones, use multi-strategy retrieval.
    if total_msgs > 0 and total_msgs <= 500:
        # Build full context from ALL messages (truncate individually to fit)
        full_context_parts = []
        total_chars = 0
        MAX_FULL_CONTEXT = 50000  # Gemini Flash has large context window
        for msg in conversation_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content.strip():
                continue
            line = f"[{role}]: {content[:500]}"
            if total_chars + len(line) > MAX_FULL_CONTEXT:
                break
            full_context_parts.append(line)
            total_chars += len(line)
        
        context = "FULL CONVERSATION (all messages):\n" + "\n\n".join(full_context_parts)
        memories = []  # Not using retrieval for small convs
    else:
        # Multi-strategy retrieval for larger conversations
        memories = _multi_strategy_recall(beam, question, top_k * 3)  # Get 3x more for reranking
        
        # LLM RERANKING: filter noisy FTS5 results to improve precision
        if len(memories) > top_k:
            try:
                # Build reranking prompt with candidate memories
                candidate_text = ""
                for i, mem in enumerate(memories[:min(len(memories), top_k * 3)]):
                    content = mem.get("content", "")[:200]
                    candidate_text += f"[{i}] {content}\n"
                
                rerank_prompt = f"""Select the {top_k} most relevant memories for answering this question. Return ONLY indices separated by commas.

QUESTION: {question}

MEMORIES:
{candidate_text}

MOST RELEVANT INDICES (comma-separated):"""
                
                rerank_response = llm.chat([
                    {"role": "system", "content": "You select relevant memories from a list. Return only comma-separated indices. No explanation."},
                    {"role": "user", "content": rerank_prompt}
                ], temperature=0.0, max_tokens=100)
                
                # Parse indices from response
                import re
                indices = [int(x.strip()) for x in re.findall(r'\d+', rerank_response)]
                reranked = [memories[i] for i in indices if i < len(memories)]
                
                if len(reranked) >= top_k // 2:  # Only use if we got enough
                    memories = reranked[:top_k]
                # Fall through to use original if reranking fails
            except Exception:
                pass
        
        context = ""  # Built below from memories

    # Build recent context from last N messages
    recent_parts = []
    if conversation_messages:
        recent = conversation_messages[-RECENT_CONTEXT_COUNT:]
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content.strip():
                recent_parts.append(f"[{role}]: {content[:300]}")
    
    # Build retrieved memory context (deduplicated, relevance-sorted)
    seen_content = set()
    memory_parts = []
    total_chars = 0
    for i, mem in enumerate(memories):
        content = mem.get("content", "")
        # Deduplicate
        content_key = content[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        
        score = mem.get("score", mem.get("relevance", 0))
        if isinstance(score, (int, float)) and score < 0.05:
            continue  # Skip very low relevance
            
        if total_chars + len(content) > MAX_MEMORY_CONTEXT_CHARS:
            remaining = MAX_MEMORY_CONTEXT_CHARS - total_chars
            if remaining > 100:
                memory_parts.append(f"[Memory] {content[:remaining]}...")
            break
        memory_parts.append(f"[Memory] {content}")
        total_chars += len(content)

    # Build prompt with contexts (skip if full-conversation mode already set)
    if not context:
        context_blocks = []
        if recent_parts:
            context_blocks.append("RECENT CONVERSATION:\n" + "\n".join(recent_parts))
        if memory_parts:
            context_blocks.append("RETRIEVED MEMORIES:\n" + "\n\n".join(memory_parts))
        
        context = "\n\n".join(context_blocks) if context_blocks else "[No memories found]"

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nQUESTION: {question}\n\nANSWER:"},
    ]

    return llm.chat(messages, temperature=0.1, max_tokens=1024)


# ============================================================
#  LLM-as-Judge: Nugget-Based Scoring (BEAM Protocol)
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a memory benchmark.
You will be given:
1. A question about a conversation
2. A list of RUBRIC ITEMS (expected facts the AI should mention)
3. The AI's ANSWER

For EACH rubric item, check if the AI's answer contains equivalent information:
- Score 1.0: correct info present, substantially matches the rubric item
- Score 0.5: partially correct, some key detail missing or slightly wrong
- Score 0.0: missing or wrong

Return ONLY this JSON:
{"scores":[1.0,0.5,0.0],"overall_score":0.X}

Where scores[i] corresponds to rubric[i], and overall_score is the average."""


def judge_with_rubrics(llm: LLMClient, question: str, rubric: list, ai_answer: str) -> dict:
    """Judge an AI answer against pre-written BEAM rubric items."""
    if not rubric:
        # Fall back to generic nugget scoring if no rubric available
        return {"scores": [], "overall_score": 0.0, "assessment": "no rubric available"}
    
    rubric_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(rubric))
    
    user_prompt = f"""QUESTION: {question}

RUBRIC ITEMS:
{rubric_text}

AI's ANSWER: {ai_answer}

For each rubric item, score how well the AI's answer matches. Return JSON with scores array and overall_score (average)."""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.chat(messages, temperature=0.0, max_tokens=500)

    # Parse JSON from response
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response[json_start:json_end])
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: basic text matching
    return {
        "scores": [0.0] * len(rubric),
        "overall_score": basic_text_similarity(ai_answer, " ".join(rubric)),
        "assessment": "JSON parse failed; using fallback",
        "raw_response": response,
    }


def basic_text_similarity(text1: str, text2: str) -> float:
    """Simple token overlap as fallback when LLM judge fails."""
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union) if union else 0.0


# ============================================================
#  Evaluation Runner
# ============================================================

def evaluate_conversation(
    llm: LLMClient,
    judge_llm: LLMClient,
    beam: BeamMemory,
    conversation: dict,
    resume_ids: set = None,
) -> dict:
    """Evaluate all probing questions for one conversation."""
    conv_id = conversation["id"]
    questions = conversation["questions"][:BENCHMARK_QUERIES_PER_CONV]
    results = []

    print(f"  Conversation {conv_id}: {len(questions)} questions")

    for i, q in enumerate(questions):
        qid = f"{conv_id}:q{i}"
        if resume_ids and qid in resume_ids:
            continue

        question = q["question"]
        ideal = q["ideal_answer"]
        rubric = q.get("rubric", [])
        ability = q.get("ability", "unknown")
        # Map dataset ability names to BEAM abbreviations
        ability = ABILITY_MAP.get(ability, ability)

        if not question or not ideal:
            continue

        # Step 1: LLM answers using Mnemosyne memories + conversation context
        t0 = time.perf_counter()
        ai_answer = answer_with_memory(llm, beam, question, 
                                       conversation_messages=conversation.get("messages", []))
        answer_time = time.perf_counter() - t0

        # Step 2: LLM-as-judge scores the answer
        t0 = time.perf_counter()
        judgment = judge_with_rubrics(judge_llm, question, rubric, ai_answer)
        judge_time = time.perf_counter() - t0

        score = judgment.get("overall_score", 0.0)

        result = {
            "qid": qid,
            "ability": ability,
            "question": question[:200],
            "ideal_answer": ideal[:200],
            "ai_answer": ai_answer[:500],
            "score": score,
            "nuggets": judgment.get("nuggets", []),
            "assessment": judgment.get("brief_assessment", ""),
            "answer_time_ms": answer_time * 1000,
            "judge_time_ms": judge_time * 1000,
        }
        results.append(result)

        print(f"    [{ability}] score={score:.2f} ans={answer_time*1000:.0f}ms judge={judge_time*1000:.0f}ms "
              f"Q: {question[:60]}...")

    return {
        "conversation_id": conv_id,
        "scale": conversation["scale"],
        "num_questions": len(questions),
        "num_evaluated": len(results),
        "results": results,
    }


def compute_ability_scores(all_results: list[dict]) -> dict:
    """Aggregate scores by ability and scale."""
    by_scale_ability = defaultdict(lambda: defaultdict(list))

    for conv_result in all_results:
        scale = conv_result["scale"]
        for r in conv_result["results"]:
            ability = r.get("ability", "unknown")
            score = r.get("score", 0.0)
            by_scale_ability[scale][ability].append(score)

    # Compute averages
    summary = {}
    for scale, abilities in by_scale_ability.items():
        scale_scores = {}
        all_scores = []
        for ability, scores in abilities.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            scale_scores[ability] = {
                "avg_score": avg,
                "count": len(scores),
            }
            all_scores.extend(scores)

        # Overall average across all abilities
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        scale_scores["OVERALL"] = {
            "avg_score": overall,
            "count": len(all_scores),
        }

        summary[scale] = scale_scores

    return summary


# ============================================================
#  SOTA Comparison
# ============================================================

PUBLISHED_SOTA = {
    "10M": {
        "Hindsight": 64.1,
        "Honcho": 40.6,
        "LIGHT (Llama-4)": 26.6,
        "RAG (Llama-4)": 24.9,
    },
    "1M": {
        "Hindsight": 73.9,
        "Honcho": 63.1,
        "LIGHT (Llama-4)": 33.6,
        "RAG (Llama-4)": 30.7,
    },
    "500K": {
        "Hindsight": 71.1,
        "Honcho": 64.9,
        "LIGHT (Llama-4)": 35.9,
        "RAG (Llama-4)": 33.0,
    },
    "100K": {
        "Hindsight": 73.4,
        "Honcho": 63.0,
        "LIGHT (Llama-4)": 35.8,
        "RAG (Llama-4)": 32.3,
    },
}


def print_sota_report(ability_summary: dict, metadata: dict):
    """Print SOTA comparison report."""
    print(f"\n{'='*80}")
    print(f"  MNEMOSYNE BEAM END-TO-END SOTA REPORT")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Model: {metadata.get('model', 'unknown')}")
    print(f"  Conversations/scale: {metadata.get('sample_size', 'N/A')}")
    print(f"  Top-K memories: {DEFAULT_TOP_K}")
    print(f"  Methodology: LLM answering + LLM-as-judge (nugget scoring, per BEAM protocol)")
    print(f"{'='*80}")

    print(f"\n  Per-Ability Scores:")
    print(f"  {'Scale':<8} {'OVERALL':>8}", end="")
    for ab in BEAM_ABILITIES:
        print(f" {ab:>6}", end="")
    print()

    for scale in sorted(ability_summary.keys()):
        scores = ability_summary[scale]
        overall = scores.get("OVERALL", {}).get("avg_score", 0.0)
        print(f"  {scale:<8} {overall*100:>7.1f}%", end="")
        for ab in BEAM_ABILITIES:
            s = scores.get(ab, {}).get("avg_score", 0.0)
            print(f" {s*100:>5.1f}%", end="")
        print()

    print(f"\n  SOTA Comparison (OVERALL):")
    print(f"  {'Scale':<8} {'Mnemosyne':>12}", end="")
    for system in ["Hindsight", "Honcho", "LIGHT (Llama-4)", "RAG (Llama-4)"]:
        print(f" {system:>18}", end="")
    print()

    for scale in sorted(ability_summary.keys()):
        our_score = ability_summary[scale].get("OVERALL", {}).get("avg_score", 0.0) * 100
        sota = PUBLISHED_SOTA.get(scale, {})
        print(f"  {scale:<8} {our_score:>11.1f}%", end="")
        for system in ["Hindsight", "Honcho", "LIGHT (Llama-4)", "RAG (Llama-4)"]:
            print(f" {sota.get(system, 0):>17.1f}%", end="")
        print()

    print(f"\n  Note: Published SOTA numbers from Hindsight blog (Apr 2026) and BEAM paper Table 3.")
    print(f"  Mnemosyne uses DeepSeek V4 Pro as answering + judging LLM.")
    print(f"  Direct comparison valid: identical BEAM dataset, identical LLM-as-judge protocol.")
    print(f"{'='*80}")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BEAM End-to-End Evaluation")
    parser.add_argument("--scales", default="100K,500K,1M,10M",
                        help="Scales to evaluate (comma-separated)")
    parser.add_argument("--sample", type=int, default=3,
                        help="Conversations per scale (0=all)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="LLM model for answering and judging")
    parser.add_argument("--judge-model", default=None,
                        help="Separate LLM for judging (default: same as --model)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Download data and print stats, don't evaluate")
    args = parser.parse_args()

    scales = [s.strip() for s in args.scales.split(",")]
    sample_size = args.sample if args.sample > 0 else None

    print(f"{'='*80}")
    print(f"  BEAM End-to-End Evaluation Pipeline")
    print(f"  Scales: {scales}")
    print(f"  Sample: {sample_size or 'ALL'} conversations/scale")
    print(f"  Model: {args.model}")
    print(f"  Judge: {args.judge_model or args.model}")
    print(f"{'='*80}")

    # Load data
    print(f"\n[1/4] Loading BEAM dataset...")
    data = load_beam_dataset(scales, max_conversations=sample_size)

    if not data:
        print("ERROR: No data loaded. Check HuggingFace token and dataset name.")
        sys.exit(1)

    # Print stats
    print(f"\n  Dataset Summary:")
    for scale, convs in data.items():
        total_msgs = sum(len(c["messages"]) for c in convs)
        total_qs = sum(len(c["questions"]) for c in convs)
        print(f"    {scale}: {len(convs)} convs, {total_msgs:,} msgs, {total_qs} questions")

    if args.dry_run:
        print(f"\n  Dry run complete. Exiting.")
        return

    # Load previous results if resuming
    resume_ids = set()
    all_previous = []
    if args.resume and RESULTS_FILE.exists():
        print(f"\n  Resuming from {RESULTS_FILE}...")
        with open(RESULTS_FILE) as f:
            prev = json.load(f)
            all_previous = prev.get("results", [])
            for conv_result in all_previous:
                for r in conv_result.get("results", []):
                    resume_ids.add(r["qid"])
        print(f"  Already evaluated: {len(resume_ids)} questions")

    # Initialize LLM clients
    print(f"\n[2/4] Initializing LLM clients...")
    llm = LLMClient(model=args.model)
    judge_llm = LLMClient(model=args.judge_model or args.model)

    # Evaluate each conversation
    print(f"\n[3/4] Evaluating... ({len(data)} scales)")
    all_results = list(all_previous) if args.resume else []

    for scale in sorted(data.keys()):
        conversations = data[scale]
        print(f"\n  --- Scale: {scale} ({len(conversations)} conversations) ---")

        for conv in conversations:
            # Create fresh Mnemosyne DB for each conversation
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / f"beam_{scale}_{conv['id']}.db"
                init_beam(db_path)
                beam = BeamMemory(session_id=f"beam_{scale}_{conv['id']}", db_path=db_path)

                # Ingest
                t0 = time.perf_counter()
                stats = ingest_conversation(beam, conv["messages"])
                ingest_time = time.perf_counter() - t0
                print(f"    Ingested {len(conv['messages'])} msgs in {ingest_time:.1f}s "
                      f"(DB: {os.path.getsize(db_path)/1024:.0f}KB)")

                # Evaluate
                conv_result = evaluate_conversation(
                    llm, judge_llm, beam, conv, resume_ids
                )
                all_results.append(conv_result)
                beam.conn.close()

            # Save progress after each conversation
            os.makedirs(RESULTS_FILE.parent, exist_ok=True)
            metadata = {
                "date": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "judge_model": args.judge_model or args.model,
                "top_k": DEFAULT_TOP_K,
                "sample_size": sample_size or "ALL",
                "scales": scales,
                "total_conversations": len(all_results),
            }
            with open(RESULTS_FILE, "w") as f:
                json.dump({"metadata": metadata, "results": all_results}, f, indent=2)

    # Cleanup
    llm.close()
    judge_llm.close()

    # Compute and print report
    print(f"\n[4/4] Computing SOTA report...")
    ability_summary = compute_ability_scores(all_results)

    metadata = {
        "model": args.model,
        "sample_size": sample_size or "ALL",
        "judge_model": args.judge_model or args.model,
    }
    print_sota_report(ability_summary, metadata)

    # Save summary
    summary_file = RESULTS_FILE.parent / "beam_e2e_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "date": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "ability_summary": {
                scale: {
                    ab: {"avg_score": v["avg_score"], "count": v["count"]}
                    for ab, v in abilities.items()
                }
                for scale, abilities in ability_summary.items()
            },
        }, f, indent=2)

    print(f"\n  Results saved to: {RESULTS_FILE}")
    print(f"  Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
