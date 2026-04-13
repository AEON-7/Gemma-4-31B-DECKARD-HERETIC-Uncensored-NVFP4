#!/usr/bin/env python3
"""Benchmark speculative decoding: 31B DECKARD + E4B drafter"""
import time
import json
import concurrent.futures
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "deckard-31b"

PROMPTS = [
    "Explain the theory of general relativity in detail.",
    "Write a Python implementation of a red-black tree with insert and delete.",
    "What are the key differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis at the molecular level.",
    "Write a comprehensive guide to deploying machine learning models in production.",
    "Explain quantum entanglement and its implications for computing.",
    "What is the history of the Internet from ARPANET to today?",
    "Describe how a modern CPU pipeline works with branch prediction.",
]


def generate(prompt, max_tokens=300):
    start = time.time()
    try:
        r = requests.post(API_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.6,
        }, timeout=120)
        data = r.json()
        elapsed = time.time() - start
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
        return {
            "elapsed": elapsed,
            "completion_tokens": completion_tokens,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "tok_per_sec": tok_per_sec,
            "finish": data["choices"][0]["finish_reason"],
        }
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start}


def run_batch(concurrency, num_requests=None):
    if num_requests is None:
        num_requests = concurrency
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

    start = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(generate, p) for p in prompts]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    wall_time = time.time() - start

    errors = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]

    if good:
        total_tokens = sum(r["completion_tokens"] for r in good)
        avg_tok_s = sum(r["tok_per_sec"] for r in good) / len(good)
        agg_tok_s = total_tokens / wall_time
        avg_latency = sum(r["elapsed"] for r in good) / len(good)
    else:
        total_tokens = avg_tok_s = agg_tok_s = avg_latency = 0

    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "errors": len(errors),
        "total_tokens": total_tokens,
        "wall_time": round(wall_time, 1),
        "avg_tok_s": round(avg_tok_s, 1),
        "agg_tok_s": round(agg_tok_s, 1),
        "avg_latency": round(avg_latency, 1),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Benchmark: DECKARD 31B + E4B Drafter (Speculative Decoding)")
    print("=" * 70)

    # Warmup
    print("\nWarmup...")
    generate("Hello", max_tokens=50)

    for conc in [1, 2, 4]:
        print(f"\n--- Concurrency: {conc} ---")
        result = run_batch(conc, num_requests=max(conc, 4))
        print(f"  Requests:      {result['requests']} ({result['errors']} errors)")
        print(f"  Wall time:     {result['wall_time']}s")
        print(f"  Total tokens:  {result['total_tokens']}")
        print(f"  Avg tok/s:     {result['avg_tok_s']} (per request)")
        print(f"  Agg tok/s:     {result['agg_tok_s']} (aggregate)")
        print(f"  Avg latency:   {result['avg_latency']}s")

    print("\n" + "=" * 70)
    print("Done.")
