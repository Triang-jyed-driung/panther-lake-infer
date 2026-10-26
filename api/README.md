# RWKV-7 OpenAI API Server

基于Qt的OpenAI兼容API服务端，支持RWKV-7模型推理。

## 特性

- **OpenAI兼容**：支持 `/v1/chat/completions`、`/v1/completions` 和 `/v1/models` 端点
- **CPU线程配置**：同步API和推理使用4个OpenMP线程，共享大核0-3；启动时自动配置，无需额外线程绑定命令
- **状态缓存**：基于token序列hash的LRU缓存，默认保存3个状态；同时保存最后一个有效prompt位置的fp16 logits，完整缓存命中可直接继续采样。缓存格式为v3，旧版缓存不兼容，读取失败时重新prefill。
- **流式响应**：支持SSE流式输出，正确处理UTF-8多字节字符
- **严格串行**：一次只处理一个请求，新请求等待当前请求完成
- **Unicode安全**：流式输出时确保UTF-8字符完整性

API仍在开发中。模型切换及并发推理不受支持；每个引擎对象终身绑定一个模型。
CPU临时缓冲区由进程共享，换模型时使用新进程和新的引擎对象。

## 构建

```bash
cd api
make
```

缓存回归测试（不需要NPU设备，使用主机内存模拟NPU状态）：

```bash
make test_state_cache
./build/test_state_cache ../spec_decode_aligned.bin
```

HTTP串行队列回归测试（无需模型或NPU，需要允许本地TCP连接）：

```bash
make test_http_server
```

## 运行

```bash
./build/rwkv_server <draft.bin> <blob> <emb.f16.bin> [port] [max_states]
```

参数：
- `draft.bin` - 草稿模型 (CPU 0.1B)
- `blob` - NPU窗口图 (7B)
- `emb.f16.bin` - 嵌入表 (fp16)
- `port` - 监听端口 (默认8080)
- `max_states` - 最大状态缓存数 (默认3)

## API端点

### 聊天完成

```
POST /v1/chat/completions
```

请求体：
```json
{
  "model": "rwkv-7",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 1.0,
  "top_p": 1.0,
  "max_tokens": 2048,
  "stream": false
}
```

### 模型列表

```
GET /v1/models
```

### 文本续写

`POST /v1/completions` 接收原始字符串 `prompt`，不添加聊天角色或模板前缀；支持 `temperature`、`top_p`、`max_tokens`、`stop` 和 `stream`。

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rwkv-7","prompt":"Once upon a time","max_tokens":128,"stream":false}'
```

响应文本位于 `choices[0].text`。`stream: true` 时返回 SSE 文本片段并以 `[DONE]` 结束。

### 健康检查

```
GET /health
```

## 使用示例

### 非流式请求

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-7",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 流式请求

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv-7",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### Python客户端

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "rwkv-7",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## 状态缓存

状态缓存基于token序列的SHA-1哈希，存储在 `/tmp/rwkv_states/` 目录下。

缓存策略：
- 最多保存3个状态（可通过命令行参数调整）
- LRU淘汰：最久未访问的状态会被删除
- 完整SHA-1作为文件名；状态文件只包含大小信息、草稿模型状态、NPU状态和末尾logits，不保存prompt token序列
- token数量仅保留在内存索引中，用于限制前缀查找范围；API usage.prompt_tokens仍正常返回
- 请求按连接接受顺序串行处理（包括流式请求），后续连接等待；不启动并发推理

## CPU核心分配

- 核心0-3：同步API和推理引擎共享4个大核

## 性能

- 0.1B模型：约240 tok/s
- 7B模型（投机解码）：约12 tok/s

## Unicode处理

流式输出时，分词器可能将多字节Unicode字符（如中文）拆分成多个token。服务端使用UTF-8缓冲区来确保：

1. **完整性检查**：只输出完整的UTF-8字符序列
2. **缓冲累积**：不完整的字节会暂存在缓冲区中
3. **延迟输出**：等到后续token补齐字节后再输出

例如，中文字符"你"（UTF-8: 0xE4 0xBD 0xA0）如果被分成3个token：
- Token 1: 0xE4 → 缓冲区累积
- Token 2: 0xBD → 缓冲区累积
- Token 3: 0xA0 → 检测到完整字符，输出"你"

## 依赖

- Qt6 Network模块
- OpenVINO 2026.3.1（NPU支持）
- 支持AVX2的x86 CPU
