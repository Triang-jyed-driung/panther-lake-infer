/*
 * http_server.h — 基于Qt的HTTP服务器
 * ====================================
 *
 * 使用QTcpServer实现简单的HTTP服务器
 * - 同步处理，严格一次一个请求
 * - 支持SSE流式响应
 * - 与同步推理共享CPU亲和性
 * - 新请求会等待当前请求完成
 */
#pragma once

#include <QObject>
#include <QTcpServer>
#include <QTcpSocket>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <functional>
#include <algorithm>
#include <QElapsedTimer>
#include <limits>

#include "openai_types.h"
#include "inference_engine.h"

class HttpServer : public QObject {
    Q_OBJECT

public:
    explicit HttpServer(InferenceEngine* engine, int port = 8080,
                       QObject *parent = nullptr)
        : QObject(parent), engine_(engine), port_(port), processing_(false) {
        server_ = new QTcpServer(this);
        connect(server_, &QTcpServer::newConnection,
                this, &HttpServer::onNewConnection);
    }

    bool start() {
        if (!server_->listen(QHostAddress::Any, port_)) {
            fprintf(stderr, "Failed to listen on port %d: %s\n",
                    port_, qPrintable(server_->errorString()));
            return false;
        }
        printf("Listening on port %d\n", port_);
        return true;
    }

    void stop() {
        server_->close();
    }

private slots:
    void onNewConnection() {
        // All request handling, including inference, stays on this thread.
        // A reentrant notification must never start another request.
        if (processing_) return;
        processing_ = true;
        server_->pauseAccepting();
        while (server_->hasPendingConnections()) {
            QTcpSocket* socket = server_->nextPendingConnection();
            QByteArray data;
            if (readRequest(socket, data)) {
                try {
                    processRequest(socket, data);
                } catch (const std::exception& e) {
                    // A streaming response may already have sent its headers.
                    // Close it rather than appending a second HTTP response.
                    fprintf(stderr, "Request failed: %s\n", e.what());
                }
            }
            // Drain the response before advancing to the next request.
            QElapsedTimer drain;
            drain.start();
            while (socket->bytesToWrite() > 0 && drain.elapsed() < 30000) {
                if (!socket->waitForBytesWritten(30000 - drain.elapsed())) break;
            }
            socket->disconnectFromHost();
            socket->deleteLater();
        }
        processing_ = false;
        server_->resumeAccepting();
    }

private:
    bool readRequest(QTcpSocket* socket, QByteArray& data) {
        // The timeout starts when this connection becomes active, not while queued.
        QElapsedTimer timer;
        timer.start();
        auto readMore = [&]() {
            if (socket->bytesAvailable() == 0) {
                const int remaining = 10000 - (int)timer.elapsed();
                if (remaining <= 0 || !socket->waitForReadyRead(remaining)) return false;
            }
            data += socket->readAll();
            return true;
        };
        int headerEnd = 0;
        while ((headerEnd = getRequestHeaderEnd(data)) == 0) {
            if (!readMore()) {
                sendError(socket, 408, "Request Timeout");
                return false;
            }
        }
        qint64 contentLength = 0;
        bool haveLength = false;
        const auto lines = data.left(headerEnd - 4).split('\n');
        for (int i = 1; i < lines.size(); ++i) {
            const QByteArray line = lines[i].trimmed();
            const int colon = line.indexOf(':');
            if (colon <= 0) {
                sendError(socket, 400, "Invalid header");
                return false;
            }
            const auto name = line.left(colon).trimmed().toLower();
            if (name == "transfer-encoding") {
                sendError(socket, 400, "Transfer-Encoding is not supported");
                return false;
            }
            if (name != "content-length") continue;
            const auto value = line.mid(colon + 1).trimmed();
            bool ok = false;
            contentLength = value.toLongLong(&ok);
            if (haveLength || !ok || value.isEmpty() ||
                !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }) ||
                contentLength > std::numeric_limits<int>::max() - headerEnd) {
                sendError(socket, 400, "Invalid Content-Length");
                return false;
            }
            haveLength = true;
        }
        while (data.size() < headerEnd + contentLength) {
            if (!readMore()) {
                sendError(socket, 408, "Request Timeout");
                return false;
            }
        }
        // One request per connection; pipelined bytes are not part of the body.
        data.truncate(headerEnd + contentLength);
        return true;
    }

    // 获取请求头结束位置
    int getRequestHeaderEnd(const QByteArray& data) {
        int pos = data.indexOf("\r\n\r\n");
        return (pos >= 0) ? pos + 4 : 0;
    }

    // 处理请求
    void processRequest(QTcpSocket* socket, const QByteArray& data) {
        // 解析HTTP请求
        QString request = QString::fromLatin1(data);
        QStringList lines = request.split("\r\n");
        if (lines.isEmpty()) {
            sendError(socket, 400, "Bad Request");
            return;
        }

        // 解析请求行
        QStringList requestLine = lines[0].split(" ");
        if (requestLine.size() < 2) {
            sendError(socket, 400, "Bad Request");
            return;
        }

        QString method = requestLine[0];
        QString path = requestLine[1];

        // 提取body
        int bodyStart = data.indexOf("\r\n\r\n");
        QByteArray body = (bodyStart >= 0) ? data.mid(bodyStart + 4) : QByteArray();

        // 路由
        if (path == "/v1/chat/completions" && method == "POST") {
            handleChatCompletions(socket, body);
        } else if (path == "/v1/models" && method == "GET") {
            handleModels(socket);
        } else if (path == "/health" || path == "/") {
            sendJsonResponse(socket, 200, {{"status", "ok"}});
        } else {
            sendError(socket, 404, "Not Found");
        }
    }

    // 处理聊天完成请求
    void handleChatCompletions(QTcpSocket* socket, const QByteArray& body) {
        // 解析JSON
        QJsonParseError error;
        QJsonDocument doc = QJsonDocument::fromJson(body, &error);
        if (error.error != QJsonParseError::NoError || !doc.isObject()) {
            sendError(socket, 400, "Invalid JSON: " + error.errorString());
            return;
        }

        ChatCompletionRequest req = ChatCompletionRequest::fromJson(doc.object());

        // 生成ID和时间戳
        QString id = generateId();
        qint64 created = QDateTime::currentSecsSinceEpoch();

        if (req.stream) {
            // 流式响应
            handleStreamResponse(socket, req, id, created);
        } else {
            // 非流式响应
            handleNonStreamResponse(socket, req, id, created);
        }
    }

    // 处理非流式响应
    void handleNonStreamResponse(QTcpSocket* socket,
                                  const ChatCompletionRequest& req,
                                  const QString& id,
                                  qint64 created) {
        // 生成回复
        auto result = engine_->generate(
            req.messages,
            req.temperature,
            req.top_p,
            req.max_tokens,
            req.stop
        );

        // 构建响应
        ChatCompletionResponse resp;
        resp.id = id;
        resp.created = created;
        resp.model = req.model;

        ChatCompletionChoice choice;
        choice.index = 0;
        choice.message.role = "assistant";
        choice.message.content = result.text;
        choice.finish_reason = result.finish_reason;
        resp.choices.push_back(choice);

        resp.usage.prompt_tokens = result.prompt_tokens;
        resp.usage.completion_tokens = result.completion_tokens;
        resp.usage.total_tokens = result.prompt_tokens + result.completion_tokens;

        // 发送响应
        sendJsonResponse(socket, 200, resp.toJson());
    }

    // 处理流式响应
    void handleStreamResponse(QTcpSocket* socket,
                               const ChatCompletionRequest& req,
                               const QString& id,
                               qint64 created) {
        // 发送SSE头
        QByteArray response = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: text/event-stream\r\n"
                             "Cache-Control: no-cache\r\n"
                             "Connection: close\r\n"
                             "Access-Control-Allow-Origin: *\r\n\r\n";
        socket->write(response);
        socket->flush();

        // 发送开始chunk
        StreamChunk startChunk;
        startChunk.id = id;
        startChunk.created = created;
        startChunk.model = req.model;

        StreamChoice startChoice;
        startChoice.index = 0;
        startChoice.delta.role = "assistant";
        startChoice.finish_reason = "";
        startChunk.choices.push_back(startChoice);

        socket->write(startChunk.toSSE().toUtf8());
        socket->flush();

        // 生成回复，通过回调发送chunk
        auto result = engine_->generate(
            req.messages,
            req.temperature,
            req.top_p,
            req.max_tokens,
            req.stop,
            [socket, id, created, &req](const QString& token) {
                StreamChunk chunk;
                chunk.id = id;
                chunk.created = created;
                chunk.model = req.model;

                StreamChoice choice;
                choice.index = 0;
                choice.delta.content = token;
                choice.finish_reason = "";
                chunk.choices.push_back(choice);

                socket->write(chunk.toSSE().toUtf8());
                socket->flush();
            }
        );

        // 发送结束chunk
        StreamChunk endChunk;
        endChunk.id = id;
        endChunk.created = created;
        endChunk.model = req.model;

        StreamChoice endChoice;
        endChoice.index = 0;
        endChoice.finish_reason = result.finish_reason;
        endChunk.choices.push_back(endChoice);

        socket->write(endChunk.toSSE().toUtf8());
        socket->write("data: [DONE]\n\n");
        socket->flush();
    }

    // 处理模型列表请求
    void handleModels(QTcpSocket* socket) {
        ModelsResponse resp;

        ModelInfo model;
        model.id = engine_->modelName();
        model.created = QDateTime::currentSecsSinceEpoch();
        resp.data.push_back(model);

        sendJsonResponse(socket, 200, resp.toJson());
    }

    // 发送JSON响应
    void sendJsonResponse(QTcpSocket* socket, int statusCode,
                          const QJsonObject& json) {
        QJsonDocument doc(json);
        QByteArray body = doc.toJson(QJsonDocument::Compact);

        QByteArray response = "HTTP/1.1 " + QString::number(statusCode).toUtf8() + " OK\r\n"
                             "Content-Type: application/json\r\n"
                             "Content-Length: " + QString::number(body.size()).toUtf8() + "\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Connection: close\r\n\r\n" + body;

        socket->write(response);
    }

    // 发送错误响应
    void sendError(QTcpSocket* socket, int statusCode, const QString& message) {
        QJsonObject error;
        error["message"] = message;
        error["type"] = "invalid_request_error";

        QJsonObject resp;
        resp["error"] = error;

        sendJsonResponse(socket, statusCode, resp);
    }

    QTcpServer* server_;
    InferenceEngine* engine_;
    int port_;
    bool processing_;  // 是否正在处理请求
};
