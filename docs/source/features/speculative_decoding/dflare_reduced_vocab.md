# Reduced-vocabulary Gemma DFlare heads

## Status

Gemma DFlare supports a full target-ID input vocabulary with a reduced draft
output head. The implementation uses the existing DFlash `d2t` offset contract:

```text
target_id[draft_id] = draft_id + d2t[draft_id]
```

The target model still verifies every proposed token, so pruning the draft
proposal vocabulary cannot change target correctness. It can reduce acceptance
when the desired target token is outside the selected subset.

The implementation consists of:

- `vllm/model_executor/models/gemma4_dflare.py`, which allocates and loads the
  mapping, keeps the full target embedding, owns the reduced LM head, and maps
  reduced greedy argmax IDs back to target IDs;
- `vllm/model_executor/models/interfaces.py::get_top_tokens`, which performs
  mapped local argmax without materializing a full target-vocabulary tensor;
- `tools/create_reduced_dflare_checkpoint.py`, which selects frequent target
  IDs, gathers tied Gemma embedding/head rows, emits `lm_head.weight` and
  `d2t`, and records corpus coverage; and
- `tools/profile_dflare_vocab.py`, which validates dimensions, mappings,
  required tokens, runtime support, and theoretical head savings.

## Why it matters for Gemma

Gemma-4-E4B uses a 262,144-token vocabulary and hidden size 2,560. Its
unsharded BF16 draft output matrix is 1.25 GiB:

- 4,096 rows: 20 MiB (98.44% smaller);
- 8,192 rows: 40 MiB (96.88% smaller);
- 16,384 rows: 80 MiB (93.75% smaller).

The input embedding remains full-sized because accepted target IDs—including
IDs outside the proposal subset—are fed back as DFlare anchor tokens.

## Create a reduced checkpoint

Start from a converted full-vocabulary vLLM DFlare checkpoint. For a tied Gemma
target, provide the compact target component directory containing
`model.language_model.embed_tokens.weight`:

```shell
python /scratch/rbasu/vllm-nightly-dflare/tools/create_reduced_dflare_checkpoint.py \
  /path/to/full-vocab-dflare-vllm \
  /path/to/dflare-vllm-vocab16k \
  --target-components /path/to/gemma-target-components \
  --tokenizer /path/to/gemma-4-E4B-it \
  --corpus /path/to/self-distilled-training.jsonl \
  --draft-vocab-size 16384 \
  --block-size 16
```

The corpus may contain `conversations`, `messages`, `text`, `turns`, or
pre-tokenized `prompt_token_ids`. Selection is deterministic:

1. count target token frequency;
2. force tokenizer special IDs, the DFlare mask ID, and explicit required IDs;
3. select the most frequent remaining IDs;
4. sort selected target IDs;
5. gather tied target embedding rows into the reduced LM head; and
6. emit signed `target_id - draft_id` offsets.

The output includes:

- `model.safetensors` with draft weights, reduced `lm_head.weight`, and `d2t`;
- `config.json` with full `vocab_size`, reduced `draft_vocab_size`, and block
  metadata;
- `draft_id_to_target_id.json`; and
- `reduced_vocab_report.json` with coverage and dimensions.

Untied targets require the real target `lm_head.weight`; do not substitute
embedding rows.

## Serve and benchmark

Use greedy local argmax so vLLM scores only reduced rows and returns mapped
target IDs directly:

```json
{
  "method": "dflare",
  "model": "/path/to/dflare-vllm-vocab16k",
  "num_speculative_tokens": 3,
  "use_local_argmax_reduction": true
}
```

The benchmark harness exposes this as:

```shell
python tools/vllm_spec_benchmark.py \
  --method dflare \
  --draft_model /path/to/dflare-vllm-vocab16k \
  --num_spec_tokens 3 \
  --execution_modes cudagraph \
  --use_local_argmax_reduction
```

For the Gemma-4-E4B public-60 experiment, a 16K head covered 97.42% of training
tokens, passed a 32-token target-identity smoke, and retained essentially the
same acceptance while materially improving throughput. Results remain
model-, corpus-, hardware-, and batch-size-specific; always compare against a
matched target-only baseline.

## Validation and failure behavior

Before publishing a reduced checkpoint, verify:

- mapping length equals `draft_vocab_size`;
- reconstructed target IDs are unique and within `vocab_size`;
- required special and mask IDs are present;
- `lm_head.weight` has shape `[draft_vocab_size, draft_hidden_size]`;
- input IDs above `draft_vocab_size` still use the full shared embedding;
- greedy outputs exactly match target-only outputs; and
- acceptance and throughput are measured on held-out prompts.

The Gemma wrapper rejects a `d2t` tensor when the config declares a full-sized
draft vocabulary. A reduced head declares `has_own_lm_head=True`, preventing
vLLM from accidentally replacing it with the full target head.

## Quantization

Draft attention and MLP layers receive vLLM's draft quantization config. The
Gemma DFlare `ParallelLMHead` is not quantized by that path. Reducing the output
head therefore addresses the dominant vocabulary matrix directly; layer
quantization remains a separate optimization that requires a compatible
checkpoint and correctness benchmark.
