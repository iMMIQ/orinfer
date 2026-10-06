# Offline quantization

Use the project uv environment for CPU tools and the fixed compiler image for GPU fitting. These codecs are experimental. Validate each checkpoint against an original BF16/FP8 reference; execution completion and quality acceptance are separate. Short scene tests establish initial quality evidence, not general benchmark performance.

`e8p.py` generates our integer basis from lattice geometry without importing an external quantizer. `e8p_gpu.py` fits original floating matrices using a fused TileLang nearest-neighbor encoder, input rotation, and two weight-only scale updates. The stored table is part of the format: indices must never be interpreted with a different basis or basis ordering.

For a pinned Flash Next source, the resumable converter streams original BF16 ranges rather than downloading the entire source checkpoint:

```bash
bash tools/quantization/run_flash_next.sh \
  --index original/model.safetensors.index.json \
  --revision SOURCE_COMMIT_SHA \
  --output artifacts/quantization/flash-next/weights --stage all
```

`--source-dir` uses local original shards instead. Routed experts use E8P plus signed block128 rotation with group-major indices ready for INT8 tile decoding. PLE embeddings use E8P with a full 160-channel rotation and 42-byte row packets; the inverse follows lookup. Other large projections use row-scaled INT8. HC, routing, normalization and state coefficients preserve source precision. Auxiliary conversion overlaps fitting/saving with at most two original source blocks, without fetching completed or unselected chunks. `--max-chunks` runs a bounded conversion check; `--layers` selects a partial expert range. Progress records exact source-range and output SHA256 hashes, validates artifacts on resume, and distinguishes expert conversion, complete weight conversion, and pending model-quality validation. This is weight-only fitting, not activation calibration. Converted experimental shards still need the Flash Next architecture loader before online inference.

Audit a conversion, including missing logical spans and exact file identities:

```bash
.venv/bin/python -m tools.model.flash_weights \
  --converted artifacts/quantization/flash-next/weights \
  --index original/model.safetensors.index.json
```

Once every text tensor is covered, publish an HF directory with globally unique safetensors keys and a standard `model.safetensors.index.json`:

```bash
.venv/bin/python -m tools.model.flash_weights \
  --converted artifacts/quantization/flash-next/weights \
  --index original/model.safetensors.index.json \
  --output artifacts/models/flash-next-e8p-a8 \
  --config original/config.json --frontend original
```

Publication preserves every payload byte. By default it consumes temporary converted shards only after verifying and recording their destination; use `--keep-temporary-shards` if a second model copy fits. The published reader in `tools/model/flash_checkpoint.py` uses config, the standard index and safetensors metadata, without conversion progress files or a community decoder. Publication is independent of model-quality acceptance and does not provide the Rust online architecture adapter.

The experimental native execution recipe consumes only this published format:

```bash
bash tools/operators/run.sh tools/model/flash_native.py \
  artifacts/quantization/flash-next/native-01 \
  --checkpoint artifacts/models/flash-next-e8p-a8 --chunk 512 --graph on
```

It uses official chat-template tokens, deterministic greedy generation, fixed-answer teacher forcing, full-distribution top3 probabilities, target NLL and complete private-state prefix restoration checks. Dense projections use row-scaled INT8; experts remain packed E8P and decode per tile. QSA supports up to the checkpoint context limit (262144): raw index keys are averaged in groups of four, normalized and rotated at each group’s first position; per-query ReLU scores select 512 blocks plus the incomplete causal tail. KV uses group-64 symmetric INT8 with FP16 scales. Index compression and the pending ring are part of private prefix state; snapshots copy only the live prefix. `--context` defaults to 262144, and `--chunk` defaults to 512 and accepts profiles through 512 tokens. `--cases zh math` bounds an initial run; `--graph off` allows a separate execution comparison. This is an offline execution and validation recipe, not the Rust online adapter. Reports distinguish execution completion from quality acceptance. `--baseline` accepts probes from an independent original BF16/FP8 run with identical token histories; community Q2 and local FFN reconstruction do not establish model quality.

Run the native long-context capacity and state check separately:

```bash
bash tools/operators/run.sh tools/model/flash_long.py \
  artifacts/quantization/flash-next/native-long \
  --checkpoint artifacts/models/flash-next-e8p-a8 \
  --context 262144 --chunk 512 \
  --lengths 2049 8192 32768 65536 131072 262136 --decode 8
```

Every input traverses all 48 layers. Each milestone checks continuation after restoring the entire prefix, including INT8 codes/scales, compressed index keys, pending keys, GDN, convolution and PLE history. This repeated-text capacity workload does not establish long-context task quality. Add `--baseline` with the original short-scene BF16 probes to run the fixed regression scenes before the capacity workload. Operator oracles and high-position checks live in `tools/operators/qsa*.py`; tensor-core attention rounds local KV/probability operands to FP16 while keeping softmax and accumulators FP32. No persistent expanded KV is kept.

The index semantics follow the checkpoint's [QSA indexer](https://github.com/sgl-project/sglang/blob/c765f8818afae5a4eaa91bc7708e99e1026330ef/python/sglang/srt/layers/attention/qsa/qsa_indexer.py). Native selection uses exact radix filtering with deterministic lower-block-ID ties. Index score workspace is bounded by the chunk size and compressed capacity, avoiding a context-by-context attention matrix.

The native recipe optionally uses the checkpoint's one-layer MTP. Convert its
weights separately, using the same original revision and E8P/A8 policy:

```bash
bash tools/quantization/run_flash_next.sh \
  --index original/model.safetensors.index.json --revision SOURCE_COMMIT_SHA \
  --output artifacts/quantization/flash-next/mtp-weights \
  --component mtp --layers 0:1 --stage all
.venv/bin/python -m tools.model.flash_weights \
  --converted artifacts/quantization/flash-next/mtp-weights \
  --index original/model.safetensors.index.json \
  --output artifacts/models/flash-next-e8p-a8-mtp \
  --config original/config.json --frontend original
bash tools/operators/run.sh tools/model/flash_native.py artifacts/flash-mtp \
  --checkpoint artifacts/models/flash-next-e8p-a8 \
  --mtp-checkpoint artifacts/models/flash-next-e8p-a8-mtp \
  --mtp-drafts 3 --chunk 512 --graph on
```

The draft shares embeddings and the output head with the target. Its fusion
normalizes the complete four-stream target HC tensor, projects each branch with
the same BF16 matrix, and adds the projected next-token embedding. Its QSA KV
also uses INT8. Greedy verification commits only the matching draft prefix and
one target token. Compact GDN verification saves keys, decay and FP32 updates
for each position, then replays the accepted prefix into the FP32 state.
Convolution, PLE and pending index histories are saved at every position.
Rejected cache tails remain outside the live cursor. `--mtp-drafts` accepts 1..7 and defaults to 3. Smaller budget/context tails
use power-of-two verification profiles, with single-token decode for the last
slot. MTP is opt-in for this offline recipe and does not add online serving or
stochastic sampling support. `tools/model/validate_flash_mtp.py` compares real
requests with MTP off, tests forced rejection and complete session restoration,
and measures draft, verification, commit and refresh together.

The default draft head selects from 65536 tokens using a deterministic vocabulary
built from project source and authored prose. The target always uses the full
vocabulary, so this changes speculation efficiency without restricting target
output. Set `--mtp-vocab-size 0` for the full draft head. `--mtp-adaptive` optionally
chooses depths 1, 3 and 7 from observed acceptance and complete round cost; fixed
depth remains the default. Small-row HC fusion, compact GDN verification and
rotation/A8 fusion are enabled by default; `--native-optimizations off` disables
these for comparison. Direct expert DP4A is available with `--direct-experts` and
is disabled by default.

`tools/model/optimize_flash_mtp.py` compares optimizations against a frozen source
tree containing the original native model and MTP controller. It checks target
outputs and private-state replay, and records draft/verify/commit/refresh CUDA
spans in separate diagnostic runs. The spans include host launch gaps; they
are not sums of individual kernel times. Use `--variants baseline selected
--prefill-chunk 512` to compare warmed target-plus-draft prefill and check the
complete prefix state against the original 128-token chunks.

Tune draft depth for code generation with authored Python, Rust and TypeScript
requests. `0` disables MTP; the sweep compares every candidate with the same
target greedy continuation and reports complete warmed decode time. Use a
smaller context for the initial broad sweep, then repeat the leading depths
at the deployment context, with longer outputs and `--thinking both` (default
reasoning effort: `xhigh`):

```bash
bash tools/operators/run.sh tools/model/tune_flash_mtp.py artifacts/flash-mtp-code \
  --checkpoint artifacts/models/flash-next-e8p-a8 \
  --mtp-checkpoint artifacts/models/flash-next-e8p-a8-mtp \
  --context 16384 --lengths 512 --drafts 0 1 2 3 4 5 6 7 \
  --decode 256 --trials 2
```

Compilation, prefill and prefix restoration are outside decode timing; drafting,
verification, accepted-state commit and draft refresh are included. Generated
code and measurements stay under the output directory. This benchmark checks
MTP regression against our target; independent BF16 and functional code-quality
evaluation remain separate.

For code requests with `xhigh` thinking enabled, use fixed depth 3. Start with
depth 5 for short direct completions and depth 7 for longer direct implementations.
These are per-request choices; the controller does not switch at the thinking
delimiter. Keep graphs enabled and
prefill chunks at 512. `--budgets 384 1536` tests both bounded and complete
continuations; EOS still ends a request before its budget. Re-tune on the actual
workload rather than choosing by acceptance rate alone. Each warmed trial also
checks exact restoration of the complete target and draft private state.

Once the original reference is complete, use a new output directory to score the same histories and explicitly query the original top3 token probabilities:

```bash
bash tools/operators/run.sh tools/model/flash_native.py \
  artifacts/quantization/flash-next/native-paired \
  --checkpoint artifacts/models/flash-next-e8p-a8 --chunk 512 --graph on \
  --max-new-tokens 256 \
  --baseline artifacts/quantization/flash-next/bf16-reference/results.json
```

Review task results, target NLL and probabilities together. A different greedy token alone is not a quality failure. Keep strict output-format failures visible, and distinguish token-budget truncation from early stopping.

Prepare fixed histories with `flash_scenes` in the compiler image. Prefetch their small original BF16 embedding and PLE rows in the CPU uv environment, then obtain an independent original-weight reference:

```bash
python -m tools.model.flash_scenes --checkpoint original --output artifacts/flash-scenes.json
.venv/bin/python -m tools.model.flash_teacher_inputs \
  --index original/model.safetensors.index.json --config original/config.json \
  --scenes artifacts/flash-scenes.json --revision SOURCE_COMMIT_SHA \
  --output artifacts/quantization/flash-next/bf16-inputs --workers 4
bash tools/model/run_flash_teacher.sh \
  --index original/model.safetensors.index.json --config original/config.json \
  --scenes artifacts/flash-scenes.json --revision SOURCE_COMMIT_SHA \
  --inputs artifacts/quantization/flash-next/bf16-inputs \
  --output artifacts/quantization/flash-next/bf16-reference
```

The input cache verifies the pinned source, config, frozen histories, payload hash and exact original BF16 values; it does not contain model predictions or calibration statistics. It uses no GPU and can run while conversion is active. The teacher launcher serializes GPU work with conversion and operator validation. Omitting `--inputs` reads the same original rows directly.

The reference CLI also supports `--device cpu` in a CPU container with the same compiler-image dependencies. It keeps original BF16 matrices, FP32 state and reductions, and uses two Torch CPU threads; it can run independently of the GPU lock. `--header-cache` reuses pinned headers from conversion. The arithmetic device is part of the resume contract; resume on the same device and use a separate output directory for another backend.

The teacher reads only selected original BF16 experts, merging consecutive IDs into bounded chunks without downloading unrouted neighbours. It uses independent Torch math with FP32 GDN state and preserves private request boundaries. Every completed layer saves exact BF16 residuals for resume, with source ranges, hashes and routing coverage. It computes the output head in FP32 using the unchanged original BF16 weights. `--stop-after-layers 1 --cases zh` bounds an initial check; it does not produce a complete-model reference. CPU checks include the Transformers delta-rule oracle, a tiny complete hybrid architecture and cached-versus-direct original inputs. Validate a GPU reference separately before using it for acceptance. Reference generation and quality acceptance remain separate.

```bash
.venv/bin/python -m tools.quantization.q2i8 \
  --weights original-bf16-matrix.npy --activations routed-training-inputs.npy \
  --output weights.safetensors
```

Input is an original floating `[N,K]` matrix. `--activations` contains actual routed `[T,K]` training samples. With activations, the default is damped second-order reconstruction; `--calibration-method diagonal` selects channel second moments. Without samples, fitting is explicitly weight-only. Do not use decoded community Q2 weights as original floating input.

`orinfer.q2i8.v1` uses standard safetensors: `indices` U8 `[N,K/4]`, `codebooks` I8 `[N,K/G,4]`, `scales` F16 `[N]`. Four adjacent indices occupy one byte, lowest channel in the lowest two bits. `G` is 64 or 128. Metadata records format, group size, logical shape and provenance. Loading validates the contract; saving refuses existing paths. Payload cost is `2 + 32/G + 16/K` bits/weight, including tables and row scales.

`Weights.gpu_layout()` losslessly transposes indices/tables to group-major storage and packs each four-entry table into a little-endian U32. This replaces the source representation when uploaded. Expanded W8 is not a persistent artifact. TileLang kernels and GPU checks are in `kernels/model/q2i8.py` and `tools/operators/q2i8{,_grouped}.py`.

`python -m tools.quantization.q2i8_ffn --help` describes local FFN calibration/evaluation. Its sample NPZ requires `activations`, `routed_ids`, `prompt_ids`, boolean `calibration_mask`, and `expert_ids` matching fixture order. Calibration and evaluation prompts must be disjoint. The report retains routing coverage and explicitly marks experts without training samples. Reference outputs use BF16-derived original weights with FP32 arithmetic; reconstruction error is not token quality or a full-model acceptance result. Record the activation producer, especially when it is itself quantized.

The second-order implementation follows the error-feedback principle of [GPTQ](https://arxiv.org/abs/2210.17323), adapted to local integer codebooks and a shared output-row scale. It preserves physical channel order and adds no runtime transform.

`vq.py` defines independent experimental integer-vector fixtures in standard safetensors. VQ4 has U8 `[N,K/4]` indices and I8 `[256,4]` tables. E8P has U16 `[N,K/8]` indices and an even-integer I8 `[256,8]` basis; sign/parity and the unit shift reconstruct the final integer coordinates exactly. Both use F16 `[N]` scales and either empty or I8 `[K]` rotation signs. `gpu_layout()` uses group-major indices and little-endian packed U32 tables. Optional U16 `[N,K/128]` patches contain one valid bit, eight signed-value bits and seven position bits, adding exactly 0.125 bit/weight; replacement values use the same row scale.

Block128 rotations are normalized input-side Hadamard transforms. The corresponding floating weight transform is mathematically equivalent before quantization; A8/FP16 boundaries add numerical error. Store/charge signs, tables, patches, scales and padding in the footprint, and include activation transformation in complete-FFN performance. These codecs do not establish full-model quality, and the E8P decoder is not a reproduction of the full QuIP# quantization/training pipeline.

`block_ldlq.py` provides regularized covariance factors and a CPU oracle for reverse block error feedback following the BlockLDLQ principle. `rotation.py` provides offline forward/inverse transforms for the Flash Next dimensions using an order20 Paley Hadamard and a power-of-two factor. A two-sided transform must restore the output coordinates before nonlinearities; its reference representation is not an online operator package. These helpers do not include QuIP# fine-tuning or full-model calibration.
