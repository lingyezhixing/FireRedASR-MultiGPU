import os
import gc
import threading
import tempfile
import shutil
import base64
from typing import List, Optional, Dict, Any
from enum import Enum
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

# 从您的项目中导入核心ASR类
from fireredasr.models.fireredasr import FireRedAsr

# --- 服务器端模型路径配置 (请确保路径正确) ---
MODEL_PATHS = {
    "FireRedASR-AED-L": "pretrained_models/FireRedASR-AED-L",
    "FireRedASR-LLM-L": "pretrained_models/FireRedASR-LLM-L"
}

# --- 使用标准的Enum类型来定义可选值 ---
class ModelType(str, Enum):
    """定义了平台管理所识别的模型名称"""
    AED = "FireRedASR-AED-L"
    LLM = "FireRedASR-LLM-L"

class LlmDtype(str, Enum):
    fp32 = "fp32"
    fp16 = "fp16"
    bf16 = "bf16"

# --- 全局模型与并发管理 ---
class ModelManager:
    """
    管理ASR模型的生命周期，确保在任何时候只有一个模型被加载到内存中，
    并在配置相同时复用该模型。此类是线程安全的。
    """
    def __init__(self):
        self._model: Optional[FireRedAsr] = None
        self._config: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def _unload_model(self):
        """安全地卸载当前模型并清理GPU缓存。"""
        if self._model is not None:
            print("正在从内存中卸载现有模型...")
            del self._model
            self._model = None
            self._config = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("模型已卸载，内存已清理。")

    def get_model(self, requested_config: Dict[str, Any]) -> FireRedAsr:
        """
        获取ASR模型实例。如果请求的配置与当前加载的配置不同，
        它会先卸载旧模型，然后再加载新模型。
        """
        with self._lock:
            current_config_str = str(sorted(self._config.items())) if self._config else None
            requested_config_str = str(sorted(requested_config.items()))

            if self._model is not None and current_config_str == requested_config_str:
                print("复用具有相同配置的缓存模型。")
                return self._model
            
            print(f"配置已更改或首次加载，正在重新加载模型...")
            self._unload_model()
            
            print(f"正在使用以下配置加载新模型: {requested_config}")
            # 将枚举成员转为字符串值以兼容原始类
            config_for_load = requested_config.copy()
            if 'llm_dtype' in config_for_load and isinstance(config_for_load['llm_dtype'], Enum):
                 config_for_load['llm_dtype'] = config_for_load['llm_dtype'].value

            self._model = FireRedAsr.from_pretrained(**config_for_load)
            self._config = requested_config
            print("新模型加载成功。")
            return self._model

model_manager = ModelManager()
transcription_lock = threading.Lock()

# --- API应用设置 ---
app = FastAPI(
    title="兼容平台的 FireRedASR API 服务",
    description="一个适配大模型管理平台的、使用 FireRedAsr 提供语音识别服务的API。",
    version="2.1.0", # 版本号更新
)

# --- 临时目录设置 ---
TEMP_DIR = "Temp"
shutil.rmtree(TEMP_DIR, ignore_errors=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Pydantic模型，用于参数校验 ---
class AudioSource(BaseModel):
    """定义单个音频文件的输入格式"""
    file_name: str = Field(..., description="原始音频文件名，用于日志和识别。")
    audio_data: str = Field(..., description="Base64编码后的音频文件内容。")

class ASRRequest(BaseModel):
    """定义ASR转录请求的完整JSON结构"""
    model: ModelType = Field(..., description="要使用的ASR模型名称。")
    audio_files: List[AudioSource] = Field(..., description="一个或多个待转录的音频文件列表。")

    # --- 模型加载参数 ---
    asr_device: str = Field("cuda:0", description="ASR编码器部分所使用的设备。")
    llm_device: str = Field("cuda:1", description="LLM部分所使用的设备 (仅当 asr_type='llm' 时有效)。")
    llm_dtype: LlmDtype = Field(LlmDtype.bf16, description="LLM使用的数据类型。")
    use_flash_attn: bool = Field(False, description="是否为LLM启用 Flash Attention 2。")
    
    # --- 通用解码参数 ---
    # MODIFICATION 1: 将 batch_size 改为可选参数，默认值为 None
    batch_size: Optional[int] = Field(None, ge=1, description="模型推理的内部批处理大小。若不传入，AED模型默认为4，LLM模型默认为1。")
    beam_size: int = Field(3, ge=1, description="解码时使用的束搜索大小 (Beam Size)。")
    decode_max_len: int = Field(0, description="解码生成的最大长度，0表示不限制。")
    
    # --- AED模型专用参数 ---
    nbest: int = Field(1, description="(AED专用) 返回的最佳假设数量。")
    softmax_smoothing: float = Field(1.0, description="(AED专用) Softmax平滑因子。")
    aed_length_penalty: float = Field(0.6, description="(AED专用) 长度惩罚系数。")
    eos_penalty: float = Field(1.0, description="(AED专用) 句末符 (EOS) 惩罚系数。")

    # --- LLM模型专用参数 ---
    decode_min_len: int = Field(0, description="(LLM专用) 解码生成的最小长度。")
    repetition_penalty: float = Field(3.0, description="(LLM专用) 重复惩罚系数。")
    llm_length_penalty: float = Field(1.0, description="(LLM专用) 长度惩罚系数。")
    temperature: float = Field(1.0, description="(LLM专用) 温度系数。")

    # --- OpenAI兼容字段，此处固定 ---
    stream: bool = Field(False, description="固定为False，不支持流式输出。")


# --- 健康检查端点 ---
@app.get("/health", summary="服务健康状态检查", tags=["管理"])
def health_check():
    return {"status": "healthy"}

# --- API 端点定义 ---
@app.post("/v1/audio/transcriptions", summary="转录一个或多个音频文件 (适配平台)", tags=["核心功能"])
def transcribe_audio(request: ASRRequest = Body(...)):
    """
    接收包含Base64编码音频的JSON请求，执行语音识别，并返回结果。
    此端点为单任务锁定，以确保模型加载和推理的稳定性。
    """
    print("一个新请求已到达，正在等待进入处理队列...")
    with transcription_lock:
        print("请求已进入处理阶段，开始执行转录任务...")
        
        if not request.audio_files:
            raise HTTPException(status_code=400, detail="请求体中的 'audio_files' 列表不能为空。")

        wav_paths_to_process = []
        request_temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        print(f"为当前请求创建了临时文件夹: {request_temp_dir}")
        
        try:
            # --- 解码Base64并保存为临时文件 ---
            for audio_source in request.audio_files:
                try:
                    audio_bytes = base64.b64decode(audio_source.audio_data)
                    file_path = os.path.join(request_temp_dir, audio_source.file_name)
                    with open(file_path, "wb") as buffer:
                        buffer.write(audio_bytes)
                    wav_paths_to_process.append(file_path)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"处理文件 '{audio_source.file_name}' 失败：无法解码Base64数据或写入文件。错误: {e}")

            # --- 映射模型名称到内部类型和路径 ---
            model_name_str = request.model.value
            internal_asr_type = "llm" if "LLM" in model_name_str else "aed"
            
            model_dir = MODEL_PATHS.get(model_name_str)
            if not model_dir:
                raise HTTPException(status_code=400, detail=f"服务器未配置模型 '{model_name_str}' 的路径。")
            if not os.path.isdir(model_dir):
                 raise HTTPException(status_code=500, detail=f"服务器错误：为 '{model_name_str}' 配置的模型路径 '{model_dir}' 不存在。")

            # --- 准备模型加载和推理的配置 ---
            model_load_config = {
                "asr_type": internal_asr_type,
                "model_dir": model_dir,
                "asr_device": request.asr_device,
                "llm_device": request.llm_device,
                "llm_dtype": request.llm_dtype,
                "use_flash_attn": request.use_flash_attn,
            }
            model = model_manager.get_model(model_load_config)
            
            transcribe_config = request.model_dump(
                exclude={"model", "audio_files", "asr_device", "llm_device", "llm_dtype", "use_flash_attn", "stream"}
            )
            
            # MODIFICATION 2: 添加动态设置 batch_size 的逻辑
            if transcribe_config['batch_size'] is None:
                print("请求中未指定 'batch_size'，将根据模型类型设置默认值。")
                if internal_asr_type == 'aed':
                    transcribe_config['batch_size'] = 4
                    print("模型类型为 AED，设置 batch_size = 4。")
                else:  # internal_asr_type == 'llm'
                    transcribe_config['batch_size'] = 1
                    print("模型类型为 LLM，设置 batch_size = 1。")
            else:
                print(f"请求中指定了 'batch_size' = {transcribe_config['batch_size']}，使用该值。")
            
            # --- 执行转录 ---
            results = model.transcribe(
                all_wav_paths=wav_paths_to_process, **transcribe_config
            )
            return results
            
        finally:
            print(f"正在清理请求的临时文件夹: {request_temp_dir}...")
            shutil.rmtree(request_temp_dir, ignore_errors=True)
            print("当前请求处理完毕，释放资源锁。")

if __name__ == "__main__":
    # 推荐使用 Gunicorn 等生产级服务器部署，但 uvicorn 用于本地测试
    uvicorn.run(app, host="127.0.0.1", port=10016)