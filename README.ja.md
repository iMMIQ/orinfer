# Orinfer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

Rust と TileLang による NVIDIA Jetson 向けテキスト・視覚言語・音声推論エンジン。

**Jetson AGX Orin 64GB / CUDA SM87** 向けに設計しています。Rust がエンジンと API を実行し、TileLang がデプロイ前に GPU カーネルを実行パッケージへコンパイルします。Qwen3.8-27B、Qwen3.8-Flash-Next、Qwen3.5-4B、Index-Translate-2B/9B/35B-A3B-preview に対応します。Qwen3-ASR-1.7B は音声文字起こしに対応します。入力の対応範囲は各モデルパッケージが定義します。

## 機能

- OpenAI Chat Completions と Ollama の chat/generate API：ストリーミング、thinking、サンプリング、関数呼び出し、JSON Schema 制約、トークン確率。
- [音声認識](docs/audio.md)：Qwen3-ASR-1.7B、Rust によるファイル文字起こしと Chat 音声入力。
- [JSONL オフライン推論](docs/batch.md)：Rust CLI による連続バッチ処理、結果の逐次保存、途中再開。
- Qwen3_5 系は E8P Q4A8/W8 混合重み、Flash Next は E8P Q2 エキスパートと Q4/W8 投影を使用。精度に敏感な処理は BF16/FP16/FP32 を維持します。
- チャンク化 prefill、リクエストごとの状態分離、prefix cache、INT8 KV、モデル設定による最大 256K トークンのコンテキスト。
- ネイティブ MTP が実測コストと受理率から草稿の深さを自動選択します。`--mtp-drafts 1..7` で上限を指定し、`0` で無効化、`auto` でパッケージの上限を使用します。
- CUDA Graph は `decode_only`（既定）、`full`、`off` の 3 モード。

## クイックスタート

実行環境は **aarch64 Linux、NVIDIA CUDA Driver、コンパイル済みエンジンと対応する実行パッケージ**です。

[Releases](https://github.com/iMMIQ/orinfer/releases) からエンジン、対応する実行パッケージ、`SHA256SUMS` をダウンロードします。検証後にエンジンを展開し、`bin/orinfer` を `PATH` に追加します。準備済みモデルは別途ダウンロードします。モデルディレクトリには設定、tokenizer、chat template、量子化 safetensors が入り、`execution_package` の digest はインストールしたパッケージと一致する必要があります。

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/matching-execution.tar.xz
orinfer validate-model /path/to/prepared-model
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

既定の待受アドレスは `[::]:8088` です。`ORINFER_API_KEY` を設定すると `/v1` と `/api` の Bearer 認証が有効になります。

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":512,"stream":true}'
```

`/health`、`/v1/models`、`/v1/chat/completions` を提供します。[サービスガイド](docs/serving.md) に Ollama の起動、画像、リクエスト既定値、[OpenCode](examples/opencode.json) / [Pi](examples/pi-models.json) との接続方法を記載しています。

## 性能

テキスト・視覚言語の 6 モデルを Jetson AGX Orin 64GB と開発版実行パッケージで **2026-10-21** に測定しました。入力は固定した英中翻訳プロンプトです。ベンチマークは単一リクエストで自動調整 MTP（草稿最大 3 個）を有効にし、複数リクエストでは自動的に MTP を無効にします。

| モデル | Prefill 512 トークン | Prefill 8K トークン | 単一 decode・MTP 有効 | 8 リクエスト decode | 最大同時数 / decode | 起動→API（秒） | 起動→最初の出力（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Index-Translate-2B | 8121.13 | 7507.67 | 171.47 | 757.39 | 128 / 2361.11 | 1.12 | 1.26 |
| Qwen3.5-4B | 3592.58 | 3416.98 | 90.71 | 360.49 | 128 / 1071.29 | 1.68 | 1.83 |
| Index-Translate-9B | 2213.52 | 2223.34 | 56.49 | 233.87 | 128 / 890.74 | 2.92 | 3.10 |
| Qwen3.8-27B | 658.87 | 598.46 | 33.64 | 67.56 | 64 / 135.89 | 6.68 | 7.05 |
| Index-Translate-35B-A3B-preview | 2233.16 | 3222.22 | 82.98 | 338.65 | 128 / 824.98 | 8.01 | 8.18 |
| Qwen3.8-Flash-Next | 925.57 | 858.11 | 56.35 | 157.25 | 32 / 282.87 | 15.79 | 15.96 |

スループットは **tok/s**、起動時間は秒で表示し、複数リクエストの列は合計スループットです。単一リクエストの入力は 512 または 8192 トークン、出力は 257 トークン固定です。単一 decode 列は入力 512 トークンの結果です。Prefill は初期化、出力ヘッド、最初の出力選択を含み、decode は最初の出力ステップを除いて計測します。最大同時数の列は 8/16/32/64/128 のうち測定を完了した最大の同時数と合計 tok/s です。複数リクエストは各入力 256 トークン、各出力 65 トークンで、全リクエストの最初の出力が揃った後の 64 トークンを計測します。

各条件で 1 回ウォームアップ後、3 回のリクエストまたは固定並行グループを測定し、中央値を採用します。`decode_only`、greedy、seed `20261002` を使用し、thinking と prefix cache は無効です。固定長診断では EOS を無視します。

起動時間は新しい Rust `serve` プロセス 3 回の中央値です。プロセス起動から `/health` の準備完了、および短い英中翻訳リクエスト（出力上限 32 トークン）の最初の空でない SSE 出力までを計測します。ファイルキャッシュを保持し、モデル読込、GPU 初期化、初回リクエスト処理を含みます。SSE 出力には複数トークンが含まれる場合があります。

[性能リファレンス](docs/performance.md) に実行パッケージの識別情報、8K decode、MTP 回数、vLLM 比較、起動、オフライン処理、品質検証の範囲を記載しています。複雑な画像翻訳の品質改善を進めています。

## 主な設定

| オプション | 既定値 | 用途 |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP 待受アドレス |
| `--default-request-params` | `{}` | JSON リクエスト既定値。明示したリクエスト値を優先 |
| `--cuda-graph` | `decode_only` | CUDA Graph モード |
| `--resident-experts` | `full` | 全エキスパートを常駐。Flash Next は SSD からのオンデマンド読み込みにも対応 |
| `--prefix-cache-mib` | `4096` | GPU prefix cache の MiB 予算。0 で無効 |
| `--max-active-requests` | `32` | アクティブなリクエスト数の上限。1–128 |
| `--max-batch-tokens` | `128` | Decode 行数予算。混合 prefill はパッケージ容量を使用 |
| `--max-speculative-tokens` | `0` | MTP 検証行数予算。0 はパッケージ容量を使用 |
| `--prefill-budget-ms` | `200` | 混合 prefill の予測実行時間の目標 |

並行処理数とコンテキスト容量は実行パッケージと空きメモリに従います。実行パッケージの既定保存先は `~/.cache/orinfer/packages` です。`ORINFER_EXECUTION_CACHE` で変更できます。全オプションは `orinfer serve --help` を参照してください。

## 開発とドキュメント

Rust がオンラインの読み込み、スケジューリング、モデル実行、API 提供を担当します。Python はオフラインの量子化、コンパイル、検証に使用します。

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
make python-env
make check
```

- [サービスガイド](docs/serving.md) · [オフライン JSONL](docs/batch.md) · [性能](docs/performance.md)。
- [ビルドとリリース](tools/build/README.md) · [実行パッケージ](docs/model-packages.md)。
- [モデルのビルド](tools/model/README.md) · [GPU カーネル](kernels/README.md) · [品質評価](tools/eval/README.md)。

詳細ガイドは現在、中国語で提供しています。

## ライセンス

**LGPL-3.0-or-later**。[LICENSE](LICENSE) と [COPYING](COPYING) を参照してください。サードパーティーのコンポーネントとモデル重みには、それぞれのライセンスが適用されます。
