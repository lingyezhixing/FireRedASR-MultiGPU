import os
import gc
import threading
import tempfile
import shutil
from typing import List, Optional, Dict, Any

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from pydantic import BaseModel, Field

# 从您的项目中导入核心ASR类
from fireredasr.models.fireredasr import FireRedAsr

# --- 全局模型与并发管理 ---

class ModelManager:
    """
    管理ASR模型的生命周期，确保在任何时候只有一个模型被加载到内存中，
    并在配置相同时复用该模型。此类是线程安全的。
    """
    def __init__(self):
        self._model: Optional[FireRedAsr] = None
        self._config: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock() # 线程锁，防止多个请求同时加载模型导致冲突

    def _unload_model(self):
        """安全地卸载当前模型并清理GPU缓存。"""
        if self._model is not None:
            print("正在从内存中卸载现有模型...")
            del self._model
            self._model = None
            self._config = None
            gc.collect() # 触发垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # 清空PyTorch的CUDA缓存
            print("模型已卸载，内存已清理。")

    def get_model(self, requested_config: Dict[str, Any]) -> FireRedAsr:
        """
        获取ASR模型实例。如果请求的配置与当前加载的配置不同，
        它会先卸载旧模型，然后再加载新模型。
        """
        with self._lock:
            # 将配置字典排序后转为字符串，以便进行稳定、可靠的比较
            current_config_str = str(sorted(self._config.items())) if self._config else None
            requested_config_str = str(sorted(requested_config.items()))

            if self._model is not None and current_config_str == requested_config_str:
                print("复用具有相同配置的缓存模型。")
                return self._model
            
            print(f"配置已更改或首次加载，正在重新加载模型...")
            self._unload_model()
            
            print(f"正在使用以下配置加载新模型: {requested_config}")
            self._model = FireRedAsr.from_pretrained(**requested_config)
            self._config = requested_config
            print("新模型加载成功。")
            return self._model

# 实例化全局唯一的模型管理器
model_manager = ModelManager()

# 创建一个全局处理锁，用于将并发请求加入队列，实现顺序处理
transcription_lock = threading.Lock()


# --- API应用设置 ---

app = FastAPI(
    title="FireRedASR API 服务",
    description="一个使用 FireRedAsr 模型提供高精度语音识别服务的API。",
    version="1.1.0", # 版本更新
)

# 用于存放上传文件的临时文件夹
TEMP_DIR = "Temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Pydantic模型，用于参数校验 ---

class ASRParams(BaseModel):
    # 模型加载相关配置
    asr_type: str = Field("llm", enum=["aed", "llm"], description="要使用的ASR模型类型。")
    model_dir: str = Field("pretrained_models/FireRedASR-LLM-L", description="预训练模型所在的目录路径。")
    asr_device: str = Field("cuda:0", description="ASR编码器部分所使用的设备。")
    llm_device: str = Field("cuda:1", description="LLM部分所使用的设备 (仅当 asr_type='llm' 时有效)。")
    llm_dtype: str = Field("bf16", enum=["fp32", "fp16", "bf16"], description="LLM使用的数据类型。")
    use_flash_attn: bool = Field(False, description="是否为LLM启用 Flash Attention 2。")

    # 推理转录相关配置
    batch_size: int = Field(1, ge=1, description="模型推理的内部批处理大小，用于控制显存使用。")
    beam_size: int = Field(3, ge=1, description="解码时使用的束搜索大小 (Beam Size)。")
    decode_max_len: int = Field(0, description="解码生成的最大长度，0表示不限制。")
    
    # AED 模型专用参数
    nbest: int = Field(1, description="(AED专用) 返回的最佳假设数量。")
    softmax_smoothing: float = Field(1.0, description="(AED专用) Softmax平滑因子。")
    aed_length_penalty: float = Field(0.6, description="(AED专用) 长度惩罚系数。")
    eos_penalty: float = Field(1.0, description="(AED专用) 句末符 (EOS) 惩罚系数。")

    # LLM 模型专用参数
    decode_min_len: int = Field(0, description="(LLM专用) 解码生成的最小长度。")
    repetition_penalty: float = Field(3.0, description="(LLM专用) 重复惩罚系数。")
    llm_length_penalty: float = Field(1.0, description="(LLM专用) 长度惩罚系数。")
    temperature: float = Field(1.0, description="(LLM专用) 温度系数。")


# --- API 端点定义 ---

@app.post("/transcribe/", summary="转录音频文件或路径 (队列处理)")
def transcribe_audio(
    params: ASRParams = Depends(),
    files: Optional[List[UploadFile]] = File(None, description="一个或多个需要转录的音频文件。"),
    paths: Optional[str] = Form(None, description="以逗号分隔的、位于服务器上的音频文件路径列表。")
):
    """
    对提供的音频进行语音到文本的转录。

    **并发处理机制**: 本接口能接收并发请求，但内部使用队列机制，
    将所有请求任务 **按顺序逐一处理**，以保证系统稳定性和资源安全。
    """
    
    # 使用 "with" 语句获取全局锁。
    # 如果锁已被占用，当前请求会在此处暂停，直到锁被释放。
    # 这有效地将所有请求放入一个队列中。
    print("一个新请求已到达，正在等待进入处理队列...")
    with transcription_lock:
        print("请求已进入处理阶段，开始执行转录任务...")
        if not files and not paths:
            raise HTTPException(status_code=400, detail="必须提供 'files' (文件上传) 或 'paths' (服务器路径) 两者之一。")

        wav_paths_to_process = []
        temp_files_to_delete = []
        is_temp_file = True

        try:
            # 步骤 1: 准备待处理的音频文件路径列表
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

            # 步骤 2: 使用模型管理器获取合适的模型实例
            model_load_config = {
                "asr_type": params.asr_type,
                "model_dir": params.model_dir,
                "asr_device": params.asr_device,
                "llm_device": params.llm_device,
                "llm_dtype": params.llm_dtype,
                "use_flash_attn": params.use_flash_attn,
            }
            model = model_manager.get_model(model_load_config)

            # 步骤 3: 准备转录所需的参数
            transcribe_config = params.dict(exclude=model_load_config.keys())

            # 步骤 4: 执行转录
            results = model.transcribe(
                all_wav_paths=wav_paths_to_process,
                **transcribe_config
            )
            return results

        finally:
            # 步骤 5: 清理临时文件 (如果是通过上传文件的方式)
            if is_temp_file and temp_files_to_delete:
                print(f"正在清理 {len(temp_files_to_delete)} 个临时文件...")
                for path in temp_files_to_delete:
                    try:
                        os.remove(path)
                    except OSError as e:
                        print(f"移除临时音频文件 {path} 时出错: {e}")
                    
                    txt_path = os.path.splitext(path)[0] + ".txt"
                    try:
                        os.remove(txt_path)
                    except OSError as e:
                        print(f"移除临时txt文件 {txt_path} 时出错: {e}")
            
            print("当前请求处理完毕，释放资源锁，队列中的下一个请求将开始处理。")


# --- 如何运行 ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)