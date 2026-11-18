# Orinfer

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md)

Rust와 TileLang으로 구현한 NVIDIA Jetson용 텍스트, 비전 언어 및 음성 추론 엔진입니다.

**Jetson AGX Orin 64GB / CUDA SM87**을 대상으로 설계했습니다. Rust가 엔진과 API를 실행하고, TileLang은 배포 전에 GPU 커널을 실행 패키지로 컴파일합니다. Qwen3.8-27B, Qwen3.8-Flash-Next, Qwen3.5-4B, Index-Translate-2B/9B/35B-A3B-preview를 지원합니다. Qwen3-ASR-1.7B는 음성 전사를 지원합니다. 입력 지원 범위는 각 모델 패키지에서 정의합니다.

## 기능

- OpenAI Chat Completions 및 Ollama chat/generate API: 스트리밍, thinking, 샘플링, 함수 호출, JSON Schema 제약, 토큰 확률.
- [음성 인식](docs/audio.md): Qwen3-ASR-1.7B, Rust 기반 파일 전사 및 Chat 오디오 입력.
- [JSONL 오프라인 추론](docs/batch.md): Rust CLI를 통한 연속 배치 처리, 결과 순차 저장, 중단 지점부터 재개.
- Qwen3_5 계열은 E8P Q4A8/W8 혼합 가중치를, Flash Next는 E8P Q2 전문가와 Q4/W8 투영을 사용합니다. 정밀도에 민감한 경로는 BF16/FP16/FP32를 유지합니다.
- 청크 prefill, 요청별 상태 격리, prefix cache, INT8 KV, 모델 설정에 따른 최대 256K 토큰 컨텍스트.
- 네이티브 MTP가 실측 비용과 수락률에 따라 초안 깊이를 자동으로 선택합니다. `--mtp-drafts 1..7`은 상한을 설정하며, `0`은 비활성화하고 `auto`는 패키지 상한을 사용합니다.
- CUDA Graph 모드: `decode_only`(기본값), `full`, `off`.

## 빠른 시작

배포 환경은 **aarch64 Linux, NVIDIA CUDA Driver, 컴파일된 엔진 및 호환 실행 패키지**로 구성합니다.

[Releases](https://github.com/iMMIQ/orinfer/releases)에서 엔진 아카이브, 호환 실행 패키지, `SHA256SUMS`를 다운로드합니다. 파일을 검증한 후 엔진을 압축 해제하고 `bin/orinfer`를 `PATH`에 추가합니다. 준비된 모델은 별도로 다운로드합니다. 모델 디렉터리에는 설정, tokenizer, chat template, 양자화 safetensors가 포함되며, `execution_package` digest가 설치된 패키지와 일치해야 합니다.

```bash
sha256sum --ignore-missing -c SHA256SUMS
orinfer package install /path/to/matching-execution.tar.xz
orinfer validate-model /path/to/prepared-model
orinfer serve /path/to/prepared-model --model qwen3.8-27b
```

기본 수신 주소는 `[::]:8088`입니다. `ORINFER_API_KEY`를 설정하면 `/v1`과 `/api`에서 Bearer 인증을 사용합니다.

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Introduce yourself in one sentence."}],"max_tokens":512,"stream":true}'
```

`/health`, `/v1/models`, `/v1/chat/completions`를 제공합니다. [서비스 가이드](docs/serving.md)에서 Ollama 배포, 이미지, 요청 기본값, [OpenCode](examples/opencode.json) / [Pi](examples/pi-models.json) 연결 방법을 확인할 수 있습니다.

## 성능

텍스트·비전 언어 6개 모델을 Jetson AGX Orin 64GB와 개발용 실행 패키지로 **2026-10-21**에 측정했습니다. 입력은 고정된 영어→중국어 번역 프롬프트입니다. 벤치마크는 단일 요청에서 적응형 MTP(초안 최대 3개)를 활성화하고, 다중 요청에서는 MTP를 자동으로 비활성화합니다.

| 모델 | Prefill 512 토큰 | Prefill 8K 토큰 | 단일 decode, MTP 활성 | 8개 요청 decode | 최대 동시 요청 / decode | 시작→API(초) | 시작→첫 출력(초) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Index-Translate-2B | 8121.13 | 7507.67 | 171.47 | 757.39 | 128 / 2361.11 | 1.12 | 1.26 |
| Qwen3.5-4B | 3592.58 | 3416.98 | 90.71 | 360.49 | 128 / 1071.29 | 1.68 | 1.83 |
| Index-Translate-9B | 2213.52 | 2223.34 | 56.49 | 233.87 | 128 / 890.74 | 2.92 | 3.10 |
| Qwen3.8-27B | 658.87 | 598.46 | 33.64 | 67.56 | 64 / 135.89 | 6.68 | 7.05 |
| Index-Translate-35B-A3B-preview | 2233.16 | 3222.22 | 82.98 | 338.65 | 128 / 824.98 | 8.01 | 8.18 |
| Qwen3.8-Flash-Next | 925.57 | 858.11 | 56.35 | 157.25 | 32 / 282.87 | 15.79 | 15.96 |

처리량은 **tok/s**, 시작 시간은 초이며 다중 요청 열은 총 처리량입니다. 단일 요청의 입력은 512 또는 8192토큰, 출력은 고정 257토큰입니다. 단일 decode 열은 512토큰 입력의 결과입니다. Prefill에는 요청 초기화, 출력 헤드, 첫 출력 선택이 포함되며 decode는 첫 출력 단계를 제외합니다. 최대 동시 요청 열은 8/16/32/64/128 중 측정을 완료한 가장 큰 동시 요청 수와 총 tok/s입니다. 다중 요청은 요청당 입력 256토큰과 출력 65토큰을 사용하고, 모든 첫 출력이 준비된 뒤 후속 64토큰을 측정합니다.

각 조건을 한 번 예열한 뒤 세 번의 요청 또는 고정 동시 요청 그룹을 측정한 중앙값입니다. `decode_only`, greedy, seed `20261002`를 사용하며 thinking과 prefix cache는 비활성화합니다. 고정 길이 진단에서는 EOS를 무시합니다.

시작 시간은 새 Rust `serve` 프로세스 3회의 중앙값입니다. 프로세스 시작부터 `/health` 준비 완료, 짧은 영어→중국어 번역 요청(출력 한도 32토큰)의 첫 비어 있지 않은 SSE 출력까지 측정합니다. 파일 캐시를 유지하며 모델 로드, GPU 초기화, 첫 요청 처리를 포함합니다. SSE 출력 하나에 여러 토큰이 포함될 수 있습니다.

[성능 자료](docs/performance.md)에 실행 패키지 식별 정보, 8K decode, MTP 횟수, vLLM 비교, 시작 시간, 오프라인 처리량, 품질 검증 범위를 기록했습니다. 복잡한 이미지 번역 품질을 지속적으로 개선하고 있습니다.

## 주요 설정

| 옵션 | 기본값 | 용도 |
| --- | --- | --- |
| `--listen` | `[::]:8088` | HTTP 수신 주소 |
| `--default-request-params` | `{}` | JSON 요청 기본값. 명시한 요청 필드 우선 |
| `--cuda-graph` | `decode_only` | CUDA Graph 모드 |
| `--resident-experts` | `full` | 전체 전문가 상주. Flash Next는 SSD 기반 용량 제한도 지원 |
| `--prefix-cache-mib` | `4096` | GPU prefix cache 예산(MiB). 0은 비활성화 |
| `--max-active-requests` | `32` | 활성 요청 상한, 1–128 |
| `--max-batch-tokens` | `128` | Decode 행 예산. 혼합 prefill은 패키지 용량 사용 |
| `--max-speculative-tokens` | `0` | MTP 검증 행 예산. 0은 패키지 용량 사용 |
| `--prefill-budget-ms` | `200` | 혼합 prefill의 예상 실행 시간 목표 |

실제 동시 처리량과 컨텍스트 용량은 실행 패키지 및 가용 메모리에 따라 결정됩니다. 실행 패키지는 기본적으로 `~/.cache/orinfer/packages`에 설치하며 `ORINFER_EXECUTION_CACHE`로 위치를 지정할 수 있습니다. 전체 옵션은 `orinfer serve --help`를 참조하세요.

## 개발 및 문서

Rust가 온라인 로딩, 스케줄링, 모델 실행, API 서비스를 담당합니다. Python은 오프라인 양자화, 컴파일, 검증에 사용합니다.

```bash
git clone https://github.com/iMMIQ/orinfer.git
cd orinfer
cargo fetch --locked
make build
make python-env
make check
```

- [서비스 가이드](docs/serving.md) · [오프라인 JSONL](docs/batch.md) · [성능](docs/performance.md).
- [빌드 및 릴리스](tools/build/README.md) · [모델 실행 패키지](docs/model-packages.md).
- [모델 빌드](tools/model/README.md) · [GPU 커널](kernels/README.md) · [품질 평가](tools/eval/README.md).

상세 가이드는 현재 중국어로 제공합니다.

## 라이선스

**LGPL-3.0-or-later**. [LICENSE](LICENSE)와 [COPYING](COPYING)를 참조하세요. 서드파티 구성 요소와 모델 가중치는 각각의 라이선스를 따릅니다.
