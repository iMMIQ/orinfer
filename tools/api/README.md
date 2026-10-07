# API验证

`smoke.py`只作为客户端连接运行中的真实模型服务，覆盖任意输入长度、EOS、SSE/usage、固定seed、stop、排队和断开后的状态隔离、错误格式，以及带整数参数的工具调用和结果回传。输出路径必须不存在；启用服务鉴权时从环境变量`ORINFER_API_KEY`读取凭据。

```bash
python3 tools/api/smoke.py --base-url http://127.0.0.1:8088/v1 \
  --output artifacts/api-smoke-results.json
```

`compatibility.py`进一步验证 nullable 参数、有符号 seed、字段定位错误、JSON/schema 与 thinking、严格/指定工具及参数增量、target logprobs、Unicode bytes 和请求隔离。服务开启 MTP 时，同一套测试覆盖其约束及概率路径；传入模型目录还会验证 token-ID bias。

```bash
python3 tools/api/compatibility.py --model qwen-flash-next \
  --tokenizer-dir /path/to/prepared-model \
  --output artifacts/api-compatibility.json
```

增加 `--long-logprobs` 会另外生成最多 512 tokens、每 token 20 个候选概率的完整响应，检查超过 512 KiB 的非流式输出；此项耗时更长。

`template_reference.py`是离线参考工具，需要Transformers。它使用原始checkpoint模板生成token IDs，对照Rust模板和分词器；包含文本、tools、工具历史和thinking四种场景。JSON对象的key顺序先与Rust一致化。

```bash
python3 tools/api/template_reference.py --tokenizer-dir /path/to/tokenizer-dir \
  --output artifacts/chat-reference.json
ORINFER_TOKENIZER_DIR=/path/to/tokenizer-dir \
ORINFER_CHAT_REFERENCE="$PWD/artifacts/chat-reference.json" \
  cargo test --offline checkpoint_matches_reference_tokens -- --ignored
```

[OpenCode配置](../../examples/opencode.json)包含本地provider与仅开放read/write的agent。将其复制为独立目录中的`opencode.json`，放入`input.txt`，从该目录运行`opencode run --pure --agent orinfer --model orinfer/qwen3.8-27b '读取input.txt并将内容写到output.txt'`。它通过客户端执行工具，再把工具结果发送回Chat API；测试目录、生成文件、会话数据库和日志都不提交源码库。

增加图片、多图、图片历史和thinking模板对照时，使用已构建的视觉manifest和本机PNG图片：

```bash
python3 tools/api/template_reference.py --tokenizer-dir /path/to/tokenizer-dir \
  --vision-model /path/to/prepared-model-dir --image /path/to/image.png \
  --output artifacts/chat-image-reference.json
ORINFER_TOKENIZER_DIR=/path/to/tokenizer-dir \
ORINFER_CHAT_REFERENCE="$PWD/artifacts/chat-image-reference.json" \
  cargo test --offline checkpoint_matches_reference_tokens -- --ignored
```

参考端由Transformers原始模板和Qwen3VLProcessor生成完整input IDs，包括其原生图片token展开；Rust端通过实际Chat请求处理路径做逐token比较。
