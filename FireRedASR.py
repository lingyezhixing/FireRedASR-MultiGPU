import os
import gc
import threading
import tempfile
import shutil
from typing import List, Optional, Dict, Any
from enum import Enum # <--- 新增: 导入Enum
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, ValidationError
# 从您的项目中导入核心ASR类
from fireredasr.models.fireredasr import FireRedAsr

# --- 服务器端模型路径配置 ---
MODEL_PATHS = {
    "aed": "pretrained_models/FireRedASR-AED-L",
    "llm": "pretrained_models/FireRedASR-LLM-L"
}

# <--- 新增: 使用标准的Enum类型来定义可选值 ---
class AsrType(str, Enum):
    aed = "aed"
    llm = "llm"

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
            # 将枚举成员转为字符串值，以确保兼容性
            config_for_load = requested_config.copy()
            config_for_load['asr_type'] = config_for_load['asr_type'].value
            config_for_load['llm_dtype'] = config_for_load['llm_dtype'].value
            self._model = FireRedAsr.from_pretrained(**config_for_load)
            self._config = requested_config
            print("新模型加载成功。")
            return self._model

model_manager = ModelManager()
transcription_lock = threading.Lock()

# --- API应用设置 ---
app = FastAPI(
    title="FireRedASR API 服务",
    description="一个使用 FireRedAsr 模型提供高精度语音识别服务的API。",
    version="1.3.1", # 版本更新，修复Pydantic警告
)
TEMP_DIR = "Temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Pydantic模型，用于参数校验 ---
class ASRParams(BaseModel):
    # <--- 修改: 使用Enum类型并移除Field中的enum参数 ---
    asr_type: AsrType = Field(AsrType.llm, description="要使用的ASR模型类型。")
    asr_device: str = Field("cuda:0", description="ASR编码器部分所使用的设备。")
    llm_device: str = Field("cuda:1", description="LLM部分所使用的设备 (仅当 asr_type='llm' 时有效)。")
    llm_dtype: LlmDtype = Field(LlmDtype.bf16, description="LLM使用的数据类型。")
    use_flash_attn: bool = Field(False, description="是否为LLM启用 Flash Attention 2。")
    batch_size: int = Field(1, ge=1, description="模型推理的内部批处理大小，用于控制显存使用。")
    beam_size: int = Field(3, ge=1, description="解码时使用的束搜索大小 (Beam Size)。")
    decode_max_len: int = Field(0, description="解码生成的最大长度，0表示不限制。")
    
    nbest: int = Field(1, description="(AED专用) 返回的最佳假设数量。")
    softmax_smoothing: float = Field(1.0, description="(AED专用) Softmax平滑因子。")
    aed_length_penalty: float = Field(0.6, description="(AED专用) 长度惩罚系数。")
    eos_penalty: float = Field(1.0, description="(AED专用) 句末符 (EOS) 惩罚系数。")
    decode_min_len: int = Field(0, description="(LLM专用) 解码生成的最小长度。")
    repetition_penalty: float = Field(3.0, description="(LLM专用) 重复惩罚系数。")
    llm_length_penalty: float = Field(1.0, description="(LLM专用) 长度惩罚系数。")
    temperature: float = Field(1.0, description="(LLM专用) 温度系数。")

# --- 健康检查端点 ---
@app.get("/health", summary="服务健康状态检查")
def health_check():
    return {"status": "healthy"}

# --- API 端点定义 ---
@app.post("/transcribe/FireRedASR", summary="转录音频文件或路径 (队列处理)")
def transcribe_audio(
    # <--- 修改: 更新Form参数的类型提示 ---
    asr_type: AsrType = Form(AsrType.llm, description="要使用的ASR模型类型。"),
    asr_device: str = Form("cuda:0", description="ASR编码器部分所使用的设备。"),
    llm_device: str = Form("cuda:1", description="LLM部分所使用的设备 (仅当 asr_type='llm' 时有效)。"),
    llm_dtype: LlmDtype = Form(LlmDtype.bf16, description="LLM使用的数据类型。"),
    use_flash_attn: bool = Form(False, description="是否为LLM启用 Flash Attention 2。"),
    batch_size: int = Form(1, ge=1, description="模型推理的内部批处理大小，用于控制显存使用。"),
    beam_size: int = Form(3, ge=1, description="解码时使用的束搜索大小 (Beam Size)。"),
    decode_max_len: int = Form(0, description="解码生成的最大长度，0表示不限制。"),
    
    nbest: int = Form(1, description="(AED专用) 返回的最佳假设数量。"),
    softmax_smoothing: float = Form(1.0, description="(AED专用) Softmax平滑因子。"),
    aed_length_penalty: float = Form(0.6, description="(AED专用) 长度惩罚系数。"),
    eos_penalty: float = Form(1.0, description="(AED专用) 句末符 (EOS) 惩罚系数。"),
    decode_min_len: int = Form(0, description="(LLM专用) 解码生成的最小长度。"),
    repetition_penalty: float = Form(3.0, description="(LLM专用) 重复惩罚系数。"),
    llm_length_penalty: float = Form(1.0, description="(LLM专用) 长度惩罚系数。"),
    temperature: float = Form(1.0, description="(LLM专用) 温度系数。"),
    
    files: Optional[List[UploadFile]] = File(None, description="一个或多个需要转录的音频文件。"),
    paths: Optional[str] = Form(None, description="以逗号分隔的、位于服务器上的音频文件路径列表。")
):
    print("一个新请求已到达，正在等待进入处理队列...")
    with transcription_lock:
        print("请求已进入处理阶段，开始执行转录任务...")
        
        try:
            params = ASRParams(
                asr_type=asr_type, asr_device=asr_device, llm_device=llm_device, 
                llm_dtype=llm_dtype, use_flash_attn=use_flash_attn, batch_size=batch_size, 
                beam_size=beam_size, decode_max_len=decode_max_len, nbest=nbest, 
                softmax_smoothing=softmax_smoothing, aed_length_penalty=aed_length_penalty,
                eos_penalty=eos_penalty, decode_min_len=decode_min_len, 
                repetition_penalty=repetition_penalty, llm_length_penalty=llm_length_penalty, 
                temperature=temperature
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        
        if not files and not paths:
            raise HTTPException(status_code=400, detail="必须提供 'files' (文件上传) 或 'paths' (服务器路径) 两者之一。")
        wav_paths_to_process = []
        temp_files_to_delete = []
        is_temp_file = True
        try:
            if files:
                is_temp_file = True
                for file in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}", dir=TEMP_DIR) as tmp:
                        shutil.copyfileobj(file.file, tmp)
                        wav_paths_to_process.append(tmp.name)
                        temp_files_to_delete.append(tmp.name)
            elif paths:
                is_temp_file = False
                wav_paths_to_process.extend([p.strip() for p in paths.split(',')])
            if not wav_paths_to_process:
                raise HTTPException(status_code=400, detail="请求中未找到有效的音频文件或路径。")
            model_dir = MODEL_PATHS.get(params.asr_type.value) # 使用 .value 获取枚举的字符串值
            if not model_dir:
                raise HTTPException(status_code=400, detail=f"服务器未配置asr_type为'{params.asr_type.value}'的模型路径。")
            if not os.path.isdir(model_dir):
                 raise HTTPException(status_code=500, detail=f"服务器错误：为'{params.asr_type.value}'配置的模型路径'{model_dir}'不存在。")
            model_load_config = {
                "asr_type": params.asr_type,
                "model_dir": model_dir,
                "asr_device": params.asr_device,
                "llm_device": params.llm_device,
                "llm_dtype": params.llm_dtype,
                "use_flash_attn": params.use_flash_attn,
            }
            model = model_manager.get_model(model_load_config)
            transcribe_config = params.model_dump(exclude={"asr_type", "asr_device", "llm_device", "llm_dtype", "use_flash_attn"})
            results = model.transcribe(
                all_wav_paths=wav_paths_to_process, **transcribe_config
            )
            return results
        finally:
            if is_temp_file and temp_files_to_delete:
                print(f"正在清理 {len(temp_files_to_delete)} 个临时文件...")
                for path in temp_files_to_delete:
                    try: os.remove(path)
                    except OSError as e: print(f"移除临时音频文件 {path} 时出错: {e}")
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    try: os.remove(txt_path)
                    except OSError as e: print(f"移除临时txt文件 {txt_path} 时出错: {e}")
            
            print("当前请求处理完毕，释放资源锁，队列中的下一个请求将开始处理。")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)