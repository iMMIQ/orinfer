# Orinfer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

**Rust** と **TileLang** で実装した、テキスト・視覚言語・音声・埋め込み・リランキング向け高性能推論エンジン。

主に **NVIDIA Jetson AGX Orin 64GB（CUDA SM87）** 向けに設計されており、バックエンド非依存の共通ランタイムによって **Huawei Ascend 910B / A2（CANN 9.2）** での DeepSeek-V4.1 Flash-1017 W4A8 TP8 実験的サービングにも対応します。

---

## 主な機能

- **OpenAI & Ollama API**：ストリーミング Chat Completions、Ollama `/api/chat`・`/api/generate`、thinking と `thinking_token_budget`、関数呼び出し、厳密な JSON Schema 制約（`llguidance`）、`logprobs`。
- **マルチモーダル & マルチタスク**：
  - **テキスト・視覚言語**：Qwen3.8-27B、Qwen3.8-Flash-Next、Qwen3.5-4B、Index-Translate-2B/9B/35B-A3B-preview、および Ascend 910B TP8 の DSv4.1 Flash-1017。
  - **音声認識**：Qwen3-ASR-1.7B。Rust ネイティブのデコード・リサンプリングにより `/v1/audio/transcriptions` と Chat `input_audio` に対応（[音声認識ガイド](docs/audio.md)）。
  - **埋め込み**：Nemotron-3.5-Embed-1B。テキスト・画像・複数画像検索を `/v1/embeddings` とオフライン CLI で提供（[埋め込みガイド](docs/embeddings.md)）。
  - **リランキング**：Qwen3-Reranker-0.6B。`/v1/rerank` とオフライン CLI を提供（[リランキングガイド](docs/reranking.md)）。
  - **JSONL オフライン推論**：連続バッチ処理、プレフィックス計画、逐次保存、途中再開に対応（[バッチ推論ガイド](docs/batch.md)）。
- **量子化と省メモリ設計**：Qwen3.5/ASR/Embed/Reranker は 2 段 E8P Q4A8/W8 混合精度、Flash Next は E8P Q2A8 エキスパート + Q4/W8 投影を採用。INT8 KV キャッシュと最大 256K コンテキストに対応（[量子化ノート](docs/quantization.md)）。
- **プロダクション向けランタイム**：チャンク化 prefill、リクエストごとの状態分離、Radix Prefix Cache とオプションの CPU KV オフロード、CUDA/ACL Graph（`decode_only`、`full`、`off`）、適応型ネイティブ MTP / DSpark 投機デコード（`--mtp-drafts 0..31` または `auto`）。

---

## 対応モデル一覧

| モデル | モダリティ / タスク | バックエンド | 重み形式 | 最大コンテキスト |
| --- | --- | --- | --- | ---: |
| **Index-Translate-2B / 9B / 35B-A3B** | テキスト・画像・複数画像 | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.5-4B** | テキスト・画像・複数画像 | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.8-27B** | テキスト・画像・複数画像 | CUDA SM87 | E8P Q4A8 / W8 | 256K |
| **Qwen3.8-Flash-Next** | テキスト・画像・複数画像 | CUDA SM87 | E8P Q2A8 エキスパート + Q4/W8 | 256K |
| **Qwen3-ASR-1.7B** | 音声文字起こし | CUDA SM87 | E8P Q4A8 / W8 + FP16 音声 | 64K |
| **Nemotron-3.5-Embed-1B** | テキスト・画像埋め込み | CUDA SM87 | E8P Q4A8 / W8 | 32K |
| **Qwen3-Reranker-0.6B** | クエリ・文書リランキング | CUDA SM87 | E8P Q4A8 / W8 | 32K |
| **DSv4.1 Flash-1017** *(実験的)* | テキスト・画像・ツール・DSpark | Ascend 910B TP8 | ネイティブ W4A8 + INT8 KV | 256K |

---

## クイックスタート（Jetson AGX Orin）

### 前提条件
- `aarch64` Linux および NVIDIA CUDA Driver
- [Releases](https://github.com/iMMIQ/orinfer/releases) から取得したコンパイル済み `orinfer` エンジンと対応する実行パッケージ
- 準備済みモデルディレクトリ（設定、tokenizer、chat template、量子化 safetensors を含み、`execution_package` digest がインストール済みパッケージと一致すること）

### インストールと起動

```bash
# 1. チェックサムを検証し、bin/orinfer を PATH に追加
sha256sum --ignore-missing -c SHA256SUMS

# 2. 対応する実行パッケージをローカルキャッシュへインストール
orinfer package install /path/to/matching-execution.tar.xz

# 3. モデルファイル・実行パッケージ・重みハッシュを検証
orinfer validate-model /path/to/prepared-model

# 4. OpenAI / Ollama 互換 HTTP サーバーを起動
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

既定の待受アドレスは `[::]:8088` です。`ORINFER_API_KEY` を設定すると `/v1` と `/api` で Bearer 認証が必須になります。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 512,
    "stream": true
  }'
```

複数モデル管理モード（モデルパスなしの `orinfer serve`）、Ollama API、画像入力、[OpenCode](examples/opencode.json) / [Pi](examples/pi-models.json) / [DeepSeek Harness](docs/deepseek-harness.md) 連携は [Serving Guide](docs/serving.md) を参照してください。

Ascend 910B / A2 環境では `orinfer devices --backend ascend` と `orinfer validate-ascend --device 0` で検証でき、本番運用手順は [DSv4.1 A2 Production Guide](docs/deepseek-v41-production.md) に記載しています。

---

## 性能概要（Jetson AGX Orin 64GB）

固定の英中翻訳ベンチマーク（**2026-11-20**、Schema 2 実行パッケージ、`decode_only`、greedy、seed `20261002`、thinking・prefix cache 無効。単一リクエストは最大 3 草稿の適応型 MTP を有効化、複数リクエストは MTP 無効）。スループットは **tok/s**、起動時間は新規 `orinfer serve` プロセス 3 回の中央値（秒、ページキャッシュ保持）です。

| モデル | Prefill 512 | Prefill 8K | 単一 Decode (MTP) | 8 並行 Decode | 最大同時数 / Decode | 起動→API（秒） | 起動→初回出力（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Index-Translate-2B | 8,100.7 | 7,551.6 | 153.8 | 759.2 | 128 / 2,475.0 | 0.95 | 1.05 |
| Qwen3.5-4B | 3,582.9 | 3,379.8 | 114.7 | 360.6 | 128 / 1,094.2 | 1.19 | 1.35 |
| Index-Translate-9B | 2,223.4 | 2,243.9 | 53.7 | 234.3 | 128 / 909.0 | 1.93 | 2.12 |
| Qwen3.8-27B | 655.2 | 596.9 | 24.6 | 61.0\* | 64 / 115.7\* | 6.08 | 6.43 |
| Index-Translate-35B-A3B | 2,204.5 | 3,206.6 | 75.7 | 350.1 | 128 / 848.8 | 7.14 | 7.40 |
| Qwen3.8-Flash-Next | 979.3 | 1,081.8 | 62.9 | 163.3 | 32 / 284.8 | 15.10 | 15.27 |

\* `Qwen3.8-27B` の複数リクエスト decode は `orinfer serve --mtp-drafts auto`（HTTP/SSE、`throughput` ポリシー）で実測、他 5 モデルはエンジン `benchmark_fixed_requests` で計測しています。

パッケージ digest、8K MTP decode、**Schema 2 + P1–P6 Mini-Weka エンドツーエンド基準（2026-11-20）**、vLLM 比較、JSONL バッチ処理、Ascend 910B3 TP8 の計測結果は [Performance Reference](docs/performance.md) を参照してください。

---

## 主な起動オプション

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP 待受アドレス |
| `--default-request-params` | `{}` | JSON 形式の Chat リクエスト既定値 |
| `--cuda-graph` / `--graph` | `decode_only` | Graph モード（`decode_only`、`full`、`off`） |
| `--mtp-drafts` | `auto` | 適応型 MTP 上限（`auto`、`0` 無効、または `1..31`） |
| `--resident-experts` | `full` | 全エキスパート常駐、または層あたり SLRU 容量 `N`（Flash Next） |
| `--prefix-cache-mib` | `4096` | デバイスごとの prefix cache 予算（MiB、`0` で無効） |
| `--kv-offload-mib` | `0` | グローバル CPU prefix 状態キャッシュ予算（MiB、`0` で無効） |
| `--max-active-requests` | `32` | 最大同時アクティブ数（`1..128`） |
| `--max-batch-tokens` | `128` | 通常 decode の行数予算（`1..128`） |
| `--max-speculative-tokens` | `0` | MTP 検証行数予算（`0` はパッケージ容量を使用） |
| `--prefill-budget-ms` | `200` | `latency` ポリシー下の混合 prefill 予測時間目標 |

実行パッケージの既定保存先は `~/.cache/orinfer/packages` です（`ORINFER_EXECUTION_CACHE` で変更可能）。全オプションは `orinfer serve --help` を参照してください。

---

## ソースからのビルド

オンライン推論エンジンと CLI は Rust でビルドし、オフライン量子化・TileLang コンパイル・検証には Python 3.12+（`uv`）を使用します。

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
make python-env
make check
```

---

## ドキュメント

全体の構成は [Documentation Hub](docs/README.md) を参照してください（詳細ガイドは英語で提供しています）：

- **サービング・クライアント**：[Serving Guide](docs/serving.md) · [Offline JSONL Batch](docs/batch.md) · [DeepSeek Harness](docs/deepseek-harness.md) · [DSv4.1 A2 Production](docs/deepseek-v41-production.md)
- **モデル・モダリティ**：[Speech Recognition](docs/audio.md) · [Embeddings](docs/embeddings.md) · [Reranking](docs/reranking.md)
- **アーキテクチャ・パッケージ**：[Model Execution Packages](docs/model-packages.md) · [Ascend A2 Build Pipeline](docs/ascend-a2-build.md) · [E8P Quantization Note](docs/quantization.md)
- **ベンチマーク・開発**：[Performance Reference](docs/performance.md) · [Development Guide](docs/development.md) · [Build & Release](tools/build/README.md) · [GPU/NPU Kernels](kernels/README.md)

---

## ライセンス

**LGPL-3.0-or-later**。[LICENSE](LICENSE)、[COPYING](COPYING)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。モデル重みとサードパーティー製コンポーネントにはそれぞれのライセンスが適用されます。
