#!/usr/bin/env python3
"""Benchmark DECKARD 31B NVFP4 on DGX Spark via OpenAI-compatible API."""

import asyncio
import aiohttp
import time
import json
import statistics
import sys

BASE_URL = "http://localhost:8000/v1"
MODEL = "deckard-31b"

PROMPTS = [
    "Explain the difference between a compiler and an interpreter in two sentences.",
    "Write a Python function that checks if a string is a palindrome.",
    "What are the three laws of thermodynamics? Be concise.",
    "Describe how a neural network learns, step by step.",
    "What is the capital of Japan and what is it known for?",
    "Explain quantum entanglement to a 10-year-old.",
    "Write a haiku about artificial intelligence.",
    "What are the pros and cons of microservices architecture?",
    "How does TCP/IP work? Brief explanation.",
    "Explain the difference between SQL and NoSQL databases.",
    "What is the significance of the Turing test?",
    "Describe the water cycle in simple terms.",
    "What are design patterns in software engineering?",
    "Explain how public key cryptography works.",
    "What is the difference between machine learning and deep learning?",
    "Write a short poem about the moon.",
]

MAX_TOKENS = 200


async def send_request(session, prompt, request_id):
    """Send a single completion request and measure timing."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }

    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{BASE_URL}/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=1800),
        ) as resp:
            data = await resp.json()
            t1 = time.perf_counter()

            if "error" in data:
                return {
                    "id": request_id,
                    "error": data["error"]["message"],
                    "latency": t1 - t0,
                }

            usage = data["usage"]
            completion_tokens = usage["completion_tokens"]
            prompt_tokens = usage["prompt_tokens"]
            latency = t1 - t0

            return {
                "id": request_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": latency,
                "tok_per_sec": completion_tokens / latency if latency > 0 else 0,
            }
    except Exception as e:
        t1 = time.perf_counter()
        return {
            "id": request_id,
            "error": str(e),
            "latency": t1 - t0,
        }


async def run_benchmark(concurrency, num_requests=None):
    """Run benchmark at a given concurrency level."""
    if num_requests is None:
        num_requests = concurrency
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

    async with aiohttp.ClientSession() as session:
        t_start = time.perf_counter()
        tasks = [
            send_request(session, prompt, i)
            for i, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        t_end = time.perf_counter()

    wall_time = t_end - t_start
    successes = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not successes:
        return {
            "concurrency": concurrency,
            "successes": 0,
            "errors": len(errors),
            "error_msg": errors[0]["error"] if errors else "unknown",
        }

    total_completion_tokens = sum(r["completion_tokens"] for r in successes)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successes)
    latencies = [r["latency"] for r in successes]
    per_req_tps = [r["tok_per_sec"] for r in successes]

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successes": len(successes),
        "errors": len(errors),
        "wall_time": wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "aggregate_tok_s": total_completion_tokens / wall_time,
        "per_request_tok_s_mean": statistics.mean(per_req_tps),
        "per_request_tok_s_median": statistics.median(per_req_tps),
        "latency_mean": statistics.mean(latencies),
        "latency_median": statistics.median(latencies),
        "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
        "latency_min": min(latencies),
        "latency_max": max(latencies),
    }


async def main():
    concurrency_levels = [1, 2, 4, 8, 16, 32]

    print("=" * 100)
    print(f"DECKARD 31B NVFP4 AWQ_FULL Benchmark — DGX Spark GB10 (FLASHINFER_CUTLASS)")
    print(f"Model: {MODEL} | Max tokens: {MAX_TOKENS} | Backend: FLASHINFER_CUTLASS (native FP4)")
    print("=" * 100)

    # Warmup with completions endpoint
    print("\nWarming up (1 request, 10 tokens)...")
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": MODEL,
            "prompt": "Hello",
            "max_tokens": 10,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        async with session.post(
            f"{BASE_URL}/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            data = await resp.json()
            t1 = time.perf_counter()
            if "error" in data:
                print(f"  Warmup FAILED: {data['error']}")
                sys.exit(1)
            toks = data["usage"]["completion_tokens"]
            print(f"  Warmup OK: {toks} tokens in {t1-t0:.1f}s ({toks/(t1-t0):.2f} tok/s)")

    print()
    print(f"{'Conc':>4} | {'Reqs':>4} | {'OK':>3} | {'Err':>3} | {'Wall(s)':>8} | "
          f"{'Agg tok/s':>10} | {'Per-req tok/s':>13} | {'Lat mean':>9} | "
          f"{'Prompt':>7} | {'Compl':>7}")
    print("-" * 110)

    all_results = []
    for conc in concurrency_levels:
        sys.stdout.write(f"  Running concurrency={conc}...")
        sys.stdout.flush()
        result = await run_benchmark(conc)
        all_results.append(result)

        if result["successes"] == 0:
            print(f"\r{conc:>4} | {conc:>4} | {result['successes']:>3} | {result['errors']:>3} | "
                  f"FAILED: {result.get('error_msg', 'unknown')}")
            continue

        r = result
        print(f"\r{r['concurrency']:>4} | {r['num_requests']:>4} | {r['successes']:>3} | "
              f"{r['errors']:>3} | {r['wall_time']:>7.1f}s | {r['aggregate_tok_s']:>9.2f}  | "
              f"{r['per_request_tok_s_mean']:>8.2f} mean  | "
              f"{r['latency_mean']:>8.1f}s | "
              f"{r['total_prompt_tokens']:>7} | {r['total_completion_tokens']:>7}")

        if conc < concurrency_levels[-1]:
            await asyncio.sleep(2)

    print("-" * 110)

    # Summary
    print("\n\nMarkdown table:")
    print("| Concurrent | Aggregate tok/s | Per-Request tok/s | Avg Latency (50 tok) |")
    print("|---:|---:|---:|---:|")
    for r in all_results:
        if r["successes"] > 0:
            print(f"| {r['concurrency']} | {r['aggregate_tok_s']:.0f} | "
                  f"{r['per_request_tok_s_mean']:.0f} | {r['latency_mean']:.1f}s |")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
