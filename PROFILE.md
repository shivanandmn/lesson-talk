# Kev on CPU: a performance profile

First published CPU latency and memory figures for the open **Kev** System One
decision models, served through `llama.cpp`'s `/v1/systemone` endpoint.

As of this measurement, no CPU latency figure for Kev-4B appeared on any
HuggingFace model card or in any issue of `jaredpalmer/kev`. Published numbers
in the ecosystem were GPU, Apple Silicon, or memory-only.

## Provenance

| | |
|---|---|
| CPU | Intel Core i7-12700K (8 P-cores + 4 E-cores, 20 threads) |
| ISA | AVX2 + AVX-VNNI; **no** AVX-512, **no** AMX |
| RAM | 45 GiB |
| GPU | present but **unused** — CPU-only build |
| OS | Linux 7.0.0-43-generic x86_64 |
| llama.cpp | commit `abca692`, `-DGGML_NATIVE=ON -DLLAMA_CURL=OFF` |
| server | `-c 8192 -ub 4096 -b 4096 -np 1 --no-warmup` |
| weights | `ggml-org/Kev-0.8B-GGUF`, `ggml-org/Kev-4B-GGUF` (Apache-2.0) |
| date | 2026-11-25 |

Checksums in `results/gguf.sha256`. Raw per-config JSON in `results/*.json`.
Harness: `bench/bench.py`, `bench/final.sh`.

Latency is wall-clock round-trip to `POST /v1/systemone` from a local client,
median of 5–10 reps after 2 warmups. State is transcript-shaped text
(speaker-labelled support-call dialogue).

## Headline

| model | quant | ms/token | one decision, 90-word window | peak RSS |
|---|---|---|---|---|
| Kev-0.8B | Q8_0 | **~2.73** | **490 ms** | 1.8 GiB |
| Kev-4B | Q4_K_M | **~12.23** | **2274 ms** | 5.6 GiB |

Kev-4B is **~4.5x** the per-token cost of Kev-0.8B. At ~82 tok/s prefill, the
4B is roughly an order of magnitude below what `llama.cpp` achieves on a dense
model of similar size — consistent with the Gated DeltaNet recurrence of the
Qwen3.5 hybrid backbone having no specialised CPU kernel path.

## 1. Cost is linear in total tokens

`ms/token` is near-constant across state length and question count:

| model | thr | words | tok | q | p50 ms | ms/tok |
|---|---|---|---|---|---|---|
| Kev-0.8B Q8_0 | 8 | 90 | 187 | 1 | 490 | 2.62 |
| Kev-0.8B Q8_0 | 8 | 90 | 509 | 3 | 1317 | 2.59 |
| Kev-0.8B Q8_0 | 8 | 280 | 431 | 1 | 1238 | 2.87 |
| Kev-0.8B Q8_0 | 8 | 280 | 1241 | 3 | 3513 | 2.83 |
| Kev-4B Q4_K_M | 8 | 90 | 187 | 1 | 2274 | 12.16 |
| Kev-4B Q4_K_M | 8 | 90 | 509 | 3 | 6332 | 12.44 |
| Kev-4B Q4_K_M | 8 | 280 | 431 | 1 | 5316 | 12.33 |
| Kev-4B Q4_K_M | 8 | 280 | 1241 | 3 | 14885 | 11.99 |

## 2. The state is re-encoded per question

For an identical 90-word state, `usage.input_tokens` returns:

| questions | input_tokens |
|---|---|
| 1 | 187 |
| 3 | 509 |
| 6 | 997 |

Each question is a separate forward row that re-reads the whole state, matching
the upstream note that "the questions of a request are answered independently".

**Budget is `questions x state_length`, not `state_length`.** This is reported
by the server, not inferred from timing.

## 3. Thread count barely matters

| model | t=8 | t=12 | t=20 |
|---|---|---|---|
| Kev-0.8B Q8_0 | 2.59–2.87 | 3.25–3.39 | 2.92–3.01 |
| Kev-4B Q4_K_M | 11.99–12.44 | 12.56–13.18 | 11.58–12.02 |

A 5–25% spread. `t=12` is mildly worst on both models; `t=20` is marginally
best on the 4B. There is no large P-core/E-core penalty to engineer around.

## 4. KV-prefix reuse is worth nothing here

`--cold` gives every repetition a unique prefix, defeating the server's
longest-common-prefix slot reuse:

| model | state | q | warm p50 | cold p50 |
|---|---|---|---|---|
| Kev-0.8B Q8_0 | 90 w | 1 | 490 | 520 |
| Kev-0.8B Q8_0 | 280 w | 3 | 3513 | 3846 |
| Kev-4B Q4_K_M | 90 w | 1 | 2274 | 2254 |
| Kev-4B Q4_K_M | 280 w | 3 | 14885 | 15654 |

Within noise (cold states carry a few extra tokens). Because every question
re-encodes the state, prefix reuse cannot amortise the dominant cost. **Do not
design around cache hits.**

## 5. Q4_K_M is faster *and* smaller than Q8_0

| quant | words | p50 ms | ms/tok |
|---|---|---|---|
| Q4_K_M | 90 | 2274 | 12.16 |
| Q8_0 | 90 | 2663 | 14.24 |
| Q4_K_M | 280 | 5316 | 12.33 |
| Q8_0 | 280 | 6270 | 14.55 |

Q4_K_M is ~17% faster at 2.9 GB vs 4.2 GB. Independent published work finds
Q4_K_M costs **0.0 accuracy points** against BF16 for Kev-4B on a
1,000-decision panel, so on CPU it is strictly the better choice.

## 6. Shortest useful decision

Single boolean question, cold:

| model | words | tok | p50 ms |
|---|---|---|---|
| Kev-0.8B Q8_0 | 25 | 96 | **234** |
| Kev-0.8B Q8_0 | 50 | 134 | 408 |
| Kev-0.8B Q8_0 | 90 | 196 | 534 |
| Kev-4B Q4_K_M | 25 | 96 | **1057** |
| Kev-4B Q4_K_M | 50 | 134 | 1670 |
| Kev-4B Q4_K_M | 90 | 196 | 2425 |

## What this means for streaming audio

Conversational speech runs ~2.5 words/s, so a 90-word window takes ~36 s of
wall clock to accumulate:

| stage | compute/window | duty cycle |
|---|---|---|
| Kev-0.8B, 1 question | 0.53 s | 1.5% |
| Kev-4B, 1 question | 2.43 s | 6.7% |

Real-time streaming decisions on a CPU with no GPU are comfortable. The
constraint is not latency but **token budget**: keep windows short and ask few
questions, because cost multiplies.

Not yet measured: contention when local speech-to-text runs concurrently on the
same cores.

## Limits of this profile

- Latency and memory only. **Says nothing about whether answers are correct** —
  that needs labelled data.
- One CPU. AVX-512/AMX parts may differ substantially, as GGML's int8 kernels
  can use wider VNNI paths.
- Synthetic transcript-shaped states, not a real corpus.
- Single client, `-np 1`, no concurrency.
