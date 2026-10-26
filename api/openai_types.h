/*
 * openai_types.h — OpenAI API 数据结构
 */
#pragma once

#include <QString>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QDateTime>
#include <vector>

// Chat Message
struct ChatMessage {
    QString role;    // "system", "user", "assistant"
    QString content;
};

// Chat Completion Request
struct ChatCompletionRequest {
    QString model;
    std::vector<ChatMessage> messages;
    double temperature = 1.0;
    double top_p = 1.0;
    int max_tokens = 2048;
    bool stream = false;
    QString stop;

    static ChatCompletionRequest fromJson(const QJsonObject& obj) {
        ChatCompletionRequest req;
        req.model = obj["model"].toString("rwkv-7");
        req.temperature = obj["temperature"].toDouble(1.0);
        req.top_p = obj["top_p"].toDouble(1.0);
        req.max_tokens = obj["max_tokens"].toInt(2048);
        req.stream = obj["stream"].toBool(false);

        if (obj.contains("stop")) {
            if (obj["stop"].isString())
                req.stop = obj["stop"].toString();
            else if (obj["stop"].isArray() && !obj["stop"].toArray().isEmpty())
                req.stop = obj["stop"].toArray().first().toString();
        }

        QJsonArray msgs = obj["messages"].toArray();
        for (const auto& m : msgs) {
            QJsonObject mo = m.toObject();
            ChatMessage cm;
            cm.role = mo["role"].toString();
            cm.content = mo["content"].toString();
            req.messages.push_back(cm);
        }
        return req;
    }
};

// Raw-prompt OpenAI text completion request.
struct CompletionRequest {
    QString model = "rwkv-7";
    QString prompt;
    double temperature = 1.0;
    double top_p = 1.0;
    int max_tokens = 2048;
    bool stream = false;
    QString stop;

    static bool fromJson(const QJsonObject& obj, CompletionRequest& req,
                         QString& error) {
        const QJsonValue promptValue = obj.value("prompt");
        if (!promptValue.isString()) {
            error = "prompt must be a string";
            return false;
        }
        req.model = obj.value("model").toString("rwkv-7");
        req.prompt = promptValue.toString();
        if (req.prompt.isEmpty()) {
            error = "prompt must not be empty";
            return false;
        }
        req.temperature = obj.value("temperature").toDouble(1.0);
        req.top_p = obj.value("top_p").toDouble(1.0);
        req.max_tokens = obj.value("max_tokens").toInt(2048);
        req.stream = obj.value("stream").toBool(false);
        const QJsonValue stopValue = obj.value("stop");
        if (stopValue.isString()) req.stop = stopValue.toString();
        else if (stopValue.isArray() && !stopValue.toArray().isEmpty())
            req.stop = stopValue.toArray().first().toString();
        return true;
    }
};

// Usage Info
struct UsageInfo {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;

    QJsonObject toJson() const {
        return {
            {"prompt_tokens", prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens", total_tokens}
        };
    }
};

// Chat Completion Choice
struct ChatCompletionChoice {
    int index = 0;
    ChatMessage message;
    QString finish_reason = "stop";

    QJsonObject toJson() const {
        QJsonObject msg;
        msg["role"] = message.role;
        msg["content"] = message.content;
        return {
            {"index", index},
            {"message", msg},
            {"finish_reason", finish_reason}
        };
    }
};

// Chat Completion Response
struct ChatCompletionResponse {
    QString id;
    QString object = "chat.completion";
    qint64 created = 0;
    QString model;
    std::vector<ChatCompletionChoice> choices;
    UsageInfo usage;

    QJsonObject toJson() const {
        QJsonArray choicesArr;
        for (const auto& c : choices)
            choicesArr.append(c.toJson());

        return {
            {"id", id},
            {"object", object},
            {"created", created},
            {"model", model},
            {"choices", choicesArr},
            {"usage", usage.toJson()}
        };
    }
};

struct CompletionChoice {
    int index = 0;
    QString text;
    QString finish_reason = "stop";

    QJsonObject toJson() const {
        return {{"index", index}, {"text", text}, {"logprobs", QJsonValue::Null},
                {"finish_reason", finish_reason}};
    }
};

struct CompletionResponse {
    QString id;
    QString object = "text_completion";
    qint64 created = 0;
    QString model;
    std::vector<CompletionChoice> choices;
    UsageInfo usage;

    QJsonObject toJson() const {
        QJsonArray choicesArr;
        for (const auto& choice : choices) choicesArr.append(choice.toJson());
        return {{"id", id}, {"object", object}, {"created", created},
                {"model", model}, {"choices", choicesArr}, {"usage", usage.toJson()}};
    }
};

// Stream Chunk Delta
struct StreamDelta {
    QString role;
    QString content;

    QJsonObject toJson() const {
        QJsonObject obj;
        if (!role.isEmpty()) obj["role"] = role;
        if (!content.isEmpty()) obj["content"] = content;
        return obj;
    }
};

// Stream Chunk Choice
struct StreamChoice {
    int index = 0;
    StreamDelta delta;
    QString finish_reason;

    QJsonObject toJson() const {
        QJsonObject obj;
        obj["index"] = index;
        obj["delta"] = delta.toJson();
        if (!finish_reason.isEmpty())
            obj["finish_reason"] = finish_reason;
        else
            obj["finish_reason"] = QJsonValue::Null;
        return obj;
    }
};

// Stream Chunk Response
struct StreamChunk {
    QString id;
    QString object = "chat.completion.chunk";
    qint64 created = 0;
    QString model;
    std::vector<StreamChoice> choices;

    QJsonObject toJson() const {
        QJsonArray choicesArr;
        for (const auto& c : choices)
            choicesArr.append(c.toJson());

        return {
            {"id", id},
            {"object", object},
            {"created", created},
            {"model", model},
            {"choices", choicesArr}
        };
    }

    QString toSSE() const {
        return "data: " + QJsonDocument(toJson()).toJson(QJsonDocument::Compact) + "\n\n";
    }
};

// Model Info
struct ModelInfo {
    QString id;
    QString object = "model";
    qint64 created = 0;
    QString owned_by = "rwkv";

    QJsonObject toJson() const {
        return {
            {"id", id},
            {"object", object},
            {"created", created},
            {"owned_by", owned_by}
        };
    }
};

// Models List Response
struct ModelsResponse {
    QString object = "list";
    std::vector<ModelInfo> data;

    QJsonObject toJson() const {
        QJsonArray dataArr;
        for (const auto& m : data)
            dataArr.append(m.toJson());
        return {
            {"object", object},
            {"data", dataArr}
        };
    }
};

// 生成唯一ID
inline QString generateId(const QString& prefix = "chatcmpl") {
    static qint64 counter = 0;
    return prefix + "-" + QString::number(QDateTime::currentMSecsSinceEpoch(), 36)
                   + "-" + QString::number(++counter, 36);
}
