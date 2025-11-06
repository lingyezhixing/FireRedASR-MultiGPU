import requests
import os
import json

# --- 配置 ---
# 请将此URL修改为您API服务的实际地址和端口
API_URL = "http://127.0.0.1:8000/transcribe/"

# 请务必将此路径修改为您想要测试的音频文件的【绝对路径】
# 在Windows上，建议使用原始字符串(r"...")来避免反斜杠问题
AUDIO_FILE_PATH = r"D:\voice-translation\FireRedASR-MultiGPU\audio_split_output_vad\gaowanying_chunk_0001.wav"

def print_test_header(title):
    """打印一个漂亮的测试标题头"""
    print("\n" + "="*80)
    print(f"🔬 开始测试: {title}")
    print("="*80)

def handle_response(response):
    """统一处理和打印API响应"""
    try:
        if response.status_code == 200:
            print("✅ 请求成功 (状态码: 200)")
            # 使用json.dumps美化输出
            pretty_json = json.dumps(response.json(), indent=2, ensure_ascii=False)
            print("服务器返回结果:")
            print(pretty_json)
        else:
            print(f"❌ 请求失败 (状态码: {response.status_code})")
            print("服务器返回错误详情:")
            print(response.text)
    except requests.exceptions.JSONDecodeError:
        print("❌ 解析响应失败，服务器可能返回了非JSON格式的内容。")
        print("原始响应内容:", response.text)


def test_1_single_file_upload_llm():
    """测试1: 上传单个文件，使用默认的LLM模型配置"""
    print_test_header("上传单个文件 (LLM, 默认参数)")
    
    with open(AUDIO_FILE_PATH, 'rb') as f:
        files = {'files': (os.path.basename(AUDIO_FILE_PATH), f, 'audio/wav')}
        print(f"准备上传文件: {AUDIO_FILE_PATH}")
        print("使用参数: 默认LLM配置")
        
        response = requests.post(API_URL, files=files)
        handle_response(response)

def test_2_single_file_upload_aed_custom():
    """测试2: 上传单个文件，切换到AED模型并传入自定义参数"""
    print_test_header("上传单个文件 (AED, 自定义参数)")
    
    with open(AUDIO_FILE_PATH, 'rb') as f:
        files = {'files': (os.path.basename(AUDIO_FILE_PATH), f, 'audio/wav')}
        
        payload = {
            'asr_type': 'aed',
            'model_dir': 'pretrained_models/FireRedASR-AED-L',
            'beam_size': 5,
            'aed_length_penalty': 0.8,
            'softmax_smoothing': 1.1
        }
        
        print(f"准备上传文件: {AUDIO_FILE_PATH}")
        print(f"使用参数: {json.dumps(payload)}")

        response = requests.post(API_URL, files=files, data=payload)
        handle_response(response)

def test_3_multiple_file_upload_batch():
    """测试3: 一次性上传多个文件(使用同一个文件模拟)，测试批处理"""
    print_test_header("上传多个文件 (LLM, 批处理)")
    
    # 为了模拟，我们将同一个文件上传两次
    files_list = [
        ('files', (f"copy1_{os.path.basename(AUDIO_FILE_PATH)}", open(AUDIO_FILE_PATH, 'rb'), 'audio/wav')),
        ('files', (f"copy2_{os.path.basename(AUDIO_FILE_PATH)}", open(AUDIO_FILE_PATH, 'rb'), 'audio/wav'))
    ]
    
    payload = {
        'batch_size': 2 # 内部处理批大小，与文件数匹配以获得最佳效率
    }
    
    print(f"准备上传 {len(files_list)} 个文件 (使用同一文件模拟)")
    print(f"使用参数: {json.dumps(payload)}")

    response = requests.post(API_URL, files=files_list, data=payload)
    
    # 关闭文件句柄
    for _, (name, f, mime) in files_list:
        f.close()
        
    handle_response(response)

def test_4_server_path_single():
    """测试4: 通过服务器上的绝对路径指定单个文件"""
    print_test_header("指定服务器路径 (单个文件)")

    payload = {
        'paths': AUDIO_FILE_PATH
    }
    
    print(f"指定服务器上的文件路径: {AUDIO_FILE_PATH}")
    print(f"使用参数: {json.dumps(payload)}")

    response = requests.post(API_URL, data=payload)
    handle_response(response)

def test_5_server_path_multiple():
    """测试5: 通过服务器上的绝对路径指定多个文件"""
    print_test_header("指定服务器路径 (多个文件)")

    # 用逗号分隔同一个路径来模拟多个文件
    multiple_paths = f"{AUDIO_FILE_PATH},{AUDIO_FILE_PATH}"
    
    payload = {
        'paths': multiple_paths,
        'batch_size': 2
    }

    print(f"指定服务器上的多个文件路径: {multiple_paths}")
    print(f"使用参数: {json.dumps(payload)}")

    response = requests.post(API_URL, data=payload)
    handle_response(response)


if __name__ == "__main__":
    # 1. 检查音频文件是否存在
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"错误: 测试音频文件未找到，请检查路径配置！")
        print(f"当前配置的路径是: {AUDIO_FILE_PATH}")
    else:
        # 2. 依次运行所有测试
        try:
            test_1_single_file_upload_llm()
            test_2_single_file_upload_aed_custom()
            test_3_multiple_file_upload_batch()
            test_4_server_path_single()
            test_5_server_path_multiple()
            print("\n" + "*"*30 + " 所有测试已执行完毕 " + "*"*30)
        except requests.exceptions.ConnectionError:
            print("\n❌ 连接错误: 无法连接到API服务器。")
            print(f"请确保您的FastAPI服务正在 {API_URL} 上运行。")