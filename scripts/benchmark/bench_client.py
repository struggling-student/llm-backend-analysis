#!/usr/bin/env python3
"""Common closed-loop load generator for OpenAI-compatible completion servers.

One client for both vLLM and llama-server: same prompts, same request body,
same timing points, same metrics. Standard library only, so the exact same
code runs against both servers with no client-library differences.

Load model (closed loop): C workers, N = waves x C requests; worker w owns
requests w, w+C, w+2C, ... and sends its next one as soon as the previous one
finishes, so C requests are in flight. With --stagger-s S, worker w sends its
first request at w*S/C, so the in-flight requests are spread over their life
cycle (prefill, early and late decode) instead of all starting in lock-step.
Because each worker owns a fixed set of requests, a wrong stagger estimate can
only move the steady-state window, never starve workers.

Steady-state window: from the moment every worker has received the first token
of its first request, to the moment the first worker finishes its last request.
Exactly C requests are in flight inside it. "Steady" requests are those whose
whole life lies inside the window; latency statistics use them.

Prompts are lists of token ids: a BOS token followed by ISL-1 pseudo-random
ordinary-vocabulary tokens, generated deterministically from (seed, ISL,
request index). Both servers therefore receive exactly ISL prompt tokens
without any tokenizer or chat-template difference, and no two requests share a
prefix. Output length is fixed by max_tokens=OSL with ignore_eos=true; the
actual counts are read back from the server's usage report and checked.

Timestamps use CLOCK_MONOTONIC (time.monotonic), which every process on the
node shares, so they line up with the telemetry sampler's samples.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import random
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

READ_TIMEOUT_S = 3600.0


def make_prompt(seed: int, isl: int, index: int, bos: int, vocab: int, space: str) -> list[int]:
    rng = random.Random(f"{space}:{seed}:{isl}:{index}")
    return [bos] + [rng.randrange(1000, vocab) for _ in range(isl - 1)]


@dataclass
class Record:
    index: int
    worker: int
    t_send: float = 0.0
    t_headers: float = 0.0
    t_first: float = 0.0
    t_end: float = 0.0
    chunk_ms: list[float] = field(default_factory=list)   # chunk arrival, ms after t_send
    n_chunks: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None


def request_body(args: argparse.Namespace, prompt: list[int], osl: int, seed: int) -> bytes:
    body: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": osl,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    if args.backend == "llamacpp":
        # The server also runs --no-cache-prompt; this is the per-request switch.
        body["cache_prompt"] = False
    return json.dumps(body, separators=(",", ":")).encode()


async def _read_line(reader: asyncio.StreamReader) -> bytes:
    return await asyncio.wait_for(reader.readline(), READ_TIMEOUT_S)


async def _body_chunks(reader: asyncio.StreamReader, headers: dict[str, str]):
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await _read_line(reader)
            if not size_line:
                return
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while (await _read_line(reader)) not in (b"\r\n", b"\n", b""):
                    pass
                return
            data = await asyncio.wait_for(reader.readexactly(size), READ_TIMEOUT_S)
            await asyncio.wait_for(reader.readexactly(2), READ_TIMEOUT_S)
            yield data
    elif "content-length" in headers:
        yield await asyncio.wait_for(reader.readexactly(int(headers["content-length"])), READ_TIMEOUT_S)
    else:
        while True:
            data = await asyncio.wait_for(reader.read(65536), READ_TIMEOUT_S)
            if not data:
                return
            yield data


async def send_one(url: Any, body: bytes, rec: Record) -> None:
    """POST a streaming completion on a fresh connection and timestamp every SSE event."""
    writer = None
    try:
        reader, writer = await asyncio.open_connection(url.hostname, url.port)
        head = (
            f"POST {url.path} HTTP/1.1\r\nHost: {url.hostname}:{url.port}\r\n"
            "Content-Type: application/json\r\nAccept: text/event-stream\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode()
        rec.t_send = time.monotonic()
        writer.write(head + body)
        await writer.drain()
        status = await _read_line(reader)
        rec.http_status = int(status.split()[1])
        headers: dict[str, str] = {}
        while True:
            line = await _read_line(reader)
            if line in (b"\r\n", b"\n", b""):
                break
            k, _, v = line.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip()
        rec.t_headers = time.monotonic()
        buf = b""
        async for data in _body_chunks(reader, headers):
            t = time.monotonic()
            buf += data.replace(b"\r\n", b"\n")
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if line.startswith(b"error"):
                        rec.error = line.decode(errors="replace")[:500]
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    msg = json.loads(payload)
                    if "error" in msg:
                        rec.error = json.dumps(msg["error"])[:500]
                    choices = msg.get("choices") or []
                    if choices:
                        ch = choices[0]
                        if ch.get("text") or ch.get("finish_reason") is None:
                            if rec.n_chunks == 0:
                                rec.t_first = t
                            rec.n_chunks += 1
                            rec.chunk_ms.append(round((t - rec.t_send) * 1e3, 3))
                        if ch.get("finish_reason"):
                            rec.finish_reason = ch["finish_reason"]
                    usage = msg.get("usage")
                    if usage:
                        rec.prompt_tokens = usage.get("prompt_tokens")
                        rec.completion_tokens = usage.get("completion_tokens")
        rec.t_end = time.monotonic()
        if rec.http_status != 200 and rec.error is None:
            rec.error = f"HTTP {rec.http_status}: {buf[:300]!r}"
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, never hidden
        rec.t_end = time.monotonic()
        rec.error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        if writer is not None:
            writer.close()


async def loop_lag_monitor(stop: asyncio.Event, out: list[float]) -> None:
    """Event-loop lag: how late a 50 ms sleep wakes up. Large values mean the
    client itself is a bottleneck and its timestamps would be inflated."""
    while not stop.is_set():
        t = time.monotonic()
        await asyncio.sleep(0.05)
        out.append(time.monotonic() - t - 0.05)


async def run_closed_loop(args: argparse.Namespace, n: int, osl: int, space: str,
                          stagger_s: float) -> tuple[list[Record], dict[str, Any]]:
    url = urlparse(args.url.rstrip("/") + "/v1/completions")
    c = args.concurrency
    records: list[Record] = []
    worker_first_send: dict[int, float] = {}
    worker_idle_at: dict[int, float] = {}
    t0 = time.monotonic()

    async def worker(w: int) -> None:
        if stagger_s > 0 and c > 1:
            await asyncio.sleep(w * stagger_s / c)
        for i in range(w, n, c):
            rec = Record(index=i, worker=w)
            prompt = make_prompt(args.seed, args.isl, i, args.bos, args.vocab, space)
            await send_one(url, request_body(args, prompt, osl, args.seed + i), rec)
            worker_first_send.setdefault(w, rec.t_send)
            records.append(rec)
        worker_idle_at[w] = time.monotonic()

    stop = asyncio.Event()
    lags: list[float] = []
    monitor = asyncio.create_task(loop_lag_monitor(stop, lags))
    await asyncio.gather(*(worker(w) for w in range(min(c, n))))
    stop.set()
    await monitor
    records.sort(key=lambda r: r.index)
    timing = {
        "t_start": t0,
        "t_stop": time.monotonic(),
        "stagger_s": stagger_s,
        "t_all_workers_started": max(worker_first_send.values()) if worker_first_send else None,
        "t_first_worker_done": min(worker_idle_at.values()) if worker_idle_at else None,
        "loop_lag_p50_ms": 1e3 * statistics.median(lags) if lags else None,
        "loop_lag_max_ms": 1e3 * max(lags) if lags else None,
    }
    return records, timing


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def dist(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": pct(values, 0.50),
        "p90": pct(values, 0.90),
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize(records: list[Record], timing: dict[str, Any], isl: int, osl: int, concurrency: int) -> dict[str, Any]:
    """Steady-state window = [all C workers have received a first token, first
    worker runs out of requests]. Exactly C requests are in flight inside it."""
    ok = [r for r in records if r.error is None and r.n_chunks > 0]
    failed = [r for r in records if r not in ok]
    first_of_worker: dict[int, Record] = {}
    for r in sorted(ok, key=lambda r: r.t_send):
        first_of_worker.setdefault(r.worker, r)
    win_start = max((r.t_first for r in first_of_worker.values()), default=None)
    last_of_worker: dict[int, Record] = {}
    for r in sorted(ok, key=lambda r: r.t_send):
        last_of_worker[r.worker] = r
    win_end = min((r.t_end for r in last_of_worker.values()), default=None)
    if concurrency == 1 and ok:
        win_start, win_end = ok[0].t_send, ok[-1].t_end
    out_tok_win = 0.0
    in_tok_win = 0
    if ok and win_start is not None and win_end is not None and win_end > win_start:
        for r in ok:
            ctoks = r.completion_tokens or r.n_chunks
            w = ctoks / r.n_chunks
            out_tok_win += w * sum(1 for ms in r.chunk_ms if win_start <= r.t_send + ms / 1e3 <= win_end)
            if win_start <= r.t_first <= win_end:
                in_tok_win += r.prompt_tokens or isl
    window = (win_end - win_start) if (win_start is not None and win_end is not None) else None
    if window is not None and window <= 0:
        window = None   # no interval with all C workers active (stagger too long); flagged by steady_window_s=None
    steady = [r for r in ok if win_start is not None and win_end is not None
              and r.t_send >= win_start and r.t_end <= win_end]
    if concurrency == 1:
        steady = ok
    # Fallback (flagged in latency_sample): every request except each worker's first.
    post_ramp = [r for r in ok if r is not first_of_worker.get(r.worker)] or ok
    lat_set = steady if len(steady) >= max(3, concurrency // 2) else post_ramp

    def per_req(rs: list[Record]) -> dict[str, Any]:
        ttft = [r.t_first - r.t_send for r in rs]
        e2e = [r.t_end - r.t_send for r in rs]
        tpot = [(r.t_end - r.t_first) / ((r.completion_tokens or r.n_chunks) - 1)
                for r in rs if (r.completion_tokens or r.n_chunks) > 1]
        itl = [(b - a) / 1e3 for r in rs for a, b in zip(r.chunk_ms, r.chunk_ms[1:])]
        return {"ttft_s": dist(ttft), "tpot_s": dist(tpot), "e2e_s": dist(e2e), "itl_s": dist(itl)}

    makespan = (max(r.t_end for r in ok) - min(r.t_send for r in ok)) if ok else None
    total_out = sum((r.completion_tokens or r.n_chunks) for r in ok)
    total_in = sum((r.prompt_tokens or isl) for r in ok)
    return {
        "requests": len(records),
        "completed": len(ok),
        "failed": len(failed),
        "errors": sorted({r.error for r in failed if r.error})[:5],
        "prompt_tokens_ok": all(r.prompt_tokens == isl for r in ok),
        "completion_tokens_ok": all(r.completion_tokens == osl for r in ok),
        "chunks_equal_tokens": all(r.n_chunks == r.completion_tokens for r in ok),
        "steady_window_s": window,
        "steady_output_tok_s": out_tok_win / window if window else None,
        "steady_input_tok_s": in_tok_win / window if window else None,
        "steady_total_tok_s": (out_tok_win + in_tok_win) / window if window else None,
        "makespan_s": makespan,
        "overall_output_tok_s": total_out / makespan if makespan else None,
        "overall_input_tok_s": total_in / makespan if makespan else None,
        "overall_request_s": len(ok) / makespan if makespan else None,
        "steady_request_s": (sum(1 for r in ok if window and win_start <= r.t_end <= win_end) / window) if window else None,
        "latency_sample": "steady" if lat_set is steady else "post_ramp",
        "steady_requests": len(steady),
        "latency": per_req(lat_set),
        "latency_all": per_req(ok),
        "window": {"start": win_start, "end": win_end},
    }


def estimate_duration(records: list[Record], osl: int) -> float:
    """Request duration at this concurrency: median TTFT + (OSL-1) x median
    inter-chunk gap. Medians, because in the warm-up all prompts are prefilled
    at once and the few long stalls this causes would dominate a mean."""
    ok = [r for r in records if r.error is None and r.n_chunks > 1]
    if not ok:
        return 0.0
    ttft = statistics.median(r.t_first - r.t_send for r in ok)
    gaps = [(b - a) / 1e3 for r in ok for a, b in zip(r.chunk_ms, r.chunk_ms[1:])]
    return ttft + (osl - 1) * statistics.median(gaps)


def manifest_sha(args: argparse.Namespace, n: int, space: str) -> str:
    h = hashlib.sha256()
    for i in range(n):
        h.update(json.dumps(make_prompt(args.seed, args.isl, i, args.bos, args.vocab, space)).encode())
    return h.hexdigest()


async def amain(args: argparse.Namespace) -> int:
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0 = time.monotonic()
    result: dict[str, Any] = {"args": vars(args).copy(), "client": "bench_client.py"}

    stagger = args.stagger_s
    if args.warmup > 0:
        w_recs, w_timing = await run_closed_loop(
            argparse.Namespace(**{**vars(args), "concurrency": min(args.concurrency, args.warmup)}),
            args.warmup, args.warmup_osl, "warmup", 0.0)
        est = estimate_duration(w_recs, args.osl)
        result["warmup"] = {
            "requests": len(w_recs),
            "failed": sum(1 for r in w_recs if r.error),
            "errors": sorted({r.error for r in w_recs if r.error})[:5],
            "estimated_request_s": est,
            "summary": summarize(w_recs, w_timing, args.isl, args.warmup_osl, min(args.concurrency, args.warmup)),
        }
        if args.stagger_auto:
            stagger = est
        if any(r.error for r in w_recs):
            print(json.dumps(result["warmup"], indent=1), file=sys.stderr)
            if args.fail_on_error:
                return 2

    if args.num_requests > 0:
        records, timing = await run_closed_loop(args, args.num_requests, args.osl, "measure", stagger)
        result["timing"] = timing
        result["prompt_manifest_sha256"] = manifest_sha(args, args.num_requests, "measure")
        result["summary"] = summarize(records, timing, args.isl, args.osl, args.concurrency)
        if args.out:
            with gzip.open(args.out, "wt") as f:
                for r in records:
                    f.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")

    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.monotonic() - wall0
    cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    result["client_resources"] = {"wall_s": wall, "cpu_s": cpu, "cpu_fraction_of_one_core": cpu / wall if wall else None,
                                  "max_rss_kib": ru1.ru_maxrss}
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(result, f, indent=1)
    s = result.get("summary") or result.get("warmup", {}).get("summary", {})
    print(json.dumps({k: s.get(k) for k in ("completed", "failed", "steady_output_tok_s", "steady_total_tok_s",
                                            "prompt_tokens_ok", "completion_tokens_ok", "chunks_equal_tokens")}))
    failed = (result.get("summary") or {}).get("failed", 0)
    return 2 if (failed and args.fail_on_error) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--backend", choices=["vllm", "llamacpp"], required=True)
    p.add_argument("--isl", type=int, required=True)
    p.add_argument("--osl", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--num-requests", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bos", type=int, default=128000)
    p.add_argument("--vocab", type=int, default=128000, help="prompt tokens are drawn from [1000, vocab)")
    p.add_argument("--warmup", type=int, default=0, help="warm-up requests before measuring (not recorded)")
    p.add_argument("--warmup-osl", type=int, default=32)
    p.add_argument("--stagger-s", type=float, default=0.0)
    p.add_argument("--stagger-auto", action="store_true", help="stagger over the warm-up's estimated request duration")
    p.add_argument("--out", help="per-request records (.jsonl.gz)")
    p.add_argument("--summary", help="summary JSON")
    p.add_argument("--fail-on-error", action="store_true")
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
