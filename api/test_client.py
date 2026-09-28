#!/usr/bin/env python3
"""
RWKV-7 API 测试客户端
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8080"

def test_health():
    """测试健康检查端点"""
    print("Testing /health...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_models():
    """测试模型列表端点"""
    print("\nTesting /v1/models...")
    try:
        resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
        return resp.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat_completion():
    """测试聊天完成端点（非流式）"""
    print("\nTesting /v1/chat/completions (non-stream)...")
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv-7",
                "messages": [
                    {"role": "user", "content": "Hello! Who are you?"}
                ],
                "temperature": 1.0,
                "max_tokens": 100
            },
            timeout=60
        )
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Response:\n{json.dumps(data, indent=2)}")
        
        if resp.status_code == 200 and "choices" in data:
            content = data["choices"][0]["message"]["content"]
            print(f"\nAssistant: {content}")
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat_completion_stream():
    """测试聊天完成端点（流式）"""
    print("\nTesting /v1/chat/completions (stream)...")
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv-7",
                "messages": [
                    {"role": "user", "content": "Count from 1 to 5."}
                ],
                "temperature": 1.0,
                "max_tokens": 100,
                "stream": True
            },
            timeout=60,
            stream=True
        )
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            print("Assistant: ", end="", flush)
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            print("\n[DONE]")
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush)
                        except json.JSONDecodeError:
                            pass
            print()
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_unicode_stream():
    """测试Unicode（中文）流式输出"""
    print("\nTesting /v1/chat/completions (Unicode stream)...")
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rwkv-7",
                "messages": [
                    {"role": "user", "content": "用中文写一首简短的诗。"}
                ],
                "temperature": 1.0,
                "max_tokens": 200,
                "stream": True
            },
            timeout=120,
            stream=True
        )
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            print("Assistant: ", end="", flush=True)
            token_count = 0
            full_text = ""
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            print("\n[DONE]")
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_text += content
                                    token_count += 1
                        except json.JSONDecodeError:
                            pass
            print(f"\nReceived {token_count} stream chunks")
            print(f"Full text length: {len(full_text)} chars")
            # 验证是否包含中文
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in full_text)
            print(f"Contains Chinese: {has_chinese}")
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 50)
    print("RWKV-7 API Test Client")
    print("=" * 50)
    
    # 测试健康检查
    if not test_health():
        print("\nHealth check failed. Is the server running?")
        sys.exit(1)
    
    # 测试模型列表
    test_models()
    
    # 测试非流式聊天
    test_chat_completion()
    
    # 测试流式聊天
    test_chat_completion_stream()
    
    # 测试Unicode流式聊天
    test_unicode_stream()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
