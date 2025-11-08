import requests
import os
import json
import time
import base64

# --- 配置 ---
API_URL = "http://127.0.0.1:8080"
# 更新为新的、兼容OpenAI风格的端点
TRANScribe_URL = API_URL + "/v1/audio/transcriptions" 
HEALTH_URL = API_URL + "/health"
AUDIO_FILE_PATH = r"audio.wav" # 请确保这个文件存在

def print_test_header(title):
    """打印一个漂亮的测试标题头"""
    print("\n" + "="*80)
    print(f"🔬 开始测试: {title}")
    print("="*80)

def handle_response(response):
    """统一处理和打印API响应"""
    try:
        if response.status_code == 200:
            print(f"✅ 请求成功 (状态码: {response.status_code})")
            pretty_json = json.dumps(response.json(), indent=2, ensure_ascii=False)
            print("服务器返回结果:")
            print(pretty_json)
        else:
            print(f"❌ 请求失败 (状态码: {response.status_code})")
            print("服务器返回错误详情:")
            # 尝试解析JSON错误体
            try:
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(response.text)
    except requests.exceptions.JSONDecodeError:
        print("❌ 解析响应失败，服务器可能返回了非JSON格式的内容。")
        print("原始响应内容:", response.text)

def wait_for_server():
    """等待服务器上线，每秒检测一次健康状态"""
    print("🔍 正在等待服务器启动...")
    for _ in range(30): # 最多等待30秒
        try:
            response = requests.get(HEALTH_URL, timeout=1)
            if response.status_code == 200:
                print("✅ 服务器已成功启动并响应健康检查！")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    print("❌ 服务器启动超时！")
    return False

def get_audio_as_base64(file_path):
    """读取音频文件并返回其Base64编码的字符串"""
    with open(file_path, 'rb') as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def test_1_single_file_default_llm():
    """测试1: 上传单个文件，使用默认的LLM模型"""
    print_test_header("上传单个文件 (使用默认模型: FireRedASR-LLM-L)")
    
    audio_b64 = get_audio_as_base64(AUDIO_FILE_PATH)
    
    payload = {
        "model": "FireRedASR-LLM-L",
        "audio_files": [
            {
                "file_name": os.path.basename(AUDIO_FILE_PATH),
                "audio_data": audio_b64
            }
        ],
        "stream": False # 显式传递固定参数
    }
    
    print(f"准备上传文件: {AUDIO_FILE_PATH}")
    print("使用参数: (服务器默认)")
    response = requests.post(TRANScribe_URL, json=payload)
    handle_response(response)

def test_2_single_file_aed_custom():
    """测试2: 上传单个文件，切换到AED模型并传入自定义参数"""
    print_test_header("上传单个文件 (指定模型: FireRedASR-AED-L, 自定义参数)")
    
    audio_b64 = get_audio_as_base64(AUDIO_FILE_PATH)

    payload = {
        "model": "FireRedASR-AED-L",
        "audio_files": [
            {
                "file_name": os.path.basename(AUDIO_FILE_PATH),
                "audio_data": audio_b64
            }
        ],
        "beam_size": 5,
        "aed_length_penalty": 0.8,
        "softmax_smoothing": 1.1,
        "batch_size": 4,
        "stream": False
    }

    print(f"准备上传文件: {AUDIO_FILE_PATH}")
    print(f"使用参数: {json.dumps({k: v for k, v in payload.items() if k != 'audio_files'})}")
    response = requests.post(TRANScribe_URL, json=payload)
    handle_response(response)

def test_3_multiple_files_batch():
    """测试3: 一次性上传多个文件(使用同一个文件模拟)，测试批处理"""
    print_test_header("上传多个文件 (批处理, 模型: FireRedASR-LLM-L)")
    
    audio_b64 = get_audio_as_base64(AUDIO_FILE_PATH)
    
    payload = {
        "model": "FireRedASR-LLM-L",
        "audio_files": [
            {"file_name": f"copy_{i+1}.wav", "audio_data": audio_b64} for i in range(5)
        ],
        "batch_size": 1, # 内部处理批次大小
        "stream": False
    }

    print(f"准备上传 {len(payload['audio_files'])} 个文件 (使用同一文件模拟)")
    print(f"使用参数: {json.dumps({k: v for k, v in payload.items() if k != 'audio_files'})}")
    response = requests.post(TRANScribe_URL, json=payload)
    handle_response(response)

def test_4_validation_error():
    """测试4: 发送一个缺少必要字段的请求，测试服务器的校验逻辑"""
    print_test_header("测试校验逻辑 (发送无效请求)")

    payload = {
        "model": "FireRedASR-LLM-L",
        # "audio_files" 字段被故意省略
        "stream": False
    }
    
    print("发送一个缺少 'audio_files' 字段的请求...")
    response = requests.post(TRANScribe_URL, json=payload)
    handle_response(response) # 预期应返回 422 Unprocessable Entity

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"错误: 测试音频文件未找到，请检查路径配置！\n当前配置的路径是: {AUDIO_FILE_PATH}")
    else:
        if wait_for_server():
            # 执行所有测试
            test_1_single_file_default_llm()
            test_2_single_file_aed_custom()
            test_3_multiple_files_batch()
            test_4_validation_error()
            
            print("\n" + "*"*30 + " 所有测试已执行完毕 " + "*"*30)