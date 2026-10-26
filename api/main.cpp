/*
 * main.cpp — RWKV-7 OpenAI API Server
 * =====================================
 *
 * 基于Qt的OpenAI兼容API服务端
 * - 同步处理，一次一个请求
 * - 同步API和推理共享4大核(0-3)，由统一CPU初始化配置
 * - 状态缓存：基于token序列hash的LRU缓存
 *
 * 用法：./rwkv_server <draft.bin> <blob> <emb.f16.bin> [port] [max_states]
 */

#include <QCoreApplication>
#include <signal.h>
#include <sched.h>

#include "http_server.h"
#include "inference_engine.h"

// 全局引擎指针（用于信号处理）
static InferenceEngine* g_engine = nullptr;

void signalHandler(int sig) {
    (void)sig;
    printf("\nShutting down...\n");
    if (g_engine) g_engine->shutdown();
    QCoreApplication::quit();
}

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);

    // 解析参数
    if (argc < 4) {
        fprintf(stderr,
            "用法: %s <draft.bin> <blob> <emb.f16.bin> [port] [max_states]\n"
            "\n"
            "OpenAI兼容API服务端\n"
            "  draft.bin    - 草稿模型 (CPU 0.1B)\n"
            "  blob         - NPU窗口图 (7B)\n"
            "  emb.f16.bin  - 嵌入表 (fp16)\n"
            "  port         - 监听端口 (默认8080)\n"
            "  max_states   - 最大状态缓存数 (默认3)\n",
            argv[0]);
        return 1;
    }

    const char* draft_path = argv[1];
    const char* blob_path  = argv[2];
    const char* emb_path   = argv[3];
    int port = argc > 4 ? atoi(argv[4]) : 8080;
    int max_states = argc > 5 ? atoi(argv[5]) : 3;

    // 初始化推理引擎（会自动绑定到核心0-3）
    printf("Initializing inference engine...\n");
    InferenceEngine engine(draft_path, blob_path, emb_path, max_states);
    g_engine = &engine;

    // HTTP处理器同步调用推理，保留引擎初始化设置的0-3核心亲和性。

    // 启动HTTP服务器
    HttpServer server(&engine, port);
    if (!server.start()) {
        fprintf(stderr, "Failed to start server on port %d\n", port);
        return 1;
    }

    printf("========================================\n");
    printf("RWKV-7 API Server\n");
    printf("========================================\n");
    printf("Server: http://localhost:%d\n", port);
    printf("API:    POST /v1/chat/completions\n");
    printf("API:    POST /v1/completions\n");
    printf("Models: GET  /v1/models\n");
    printf("Health: GET  /health\n");
    printf("========================================\n\n");

    // 信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    return app.exec();
}
