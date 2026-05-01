import unsloth; print('Unsloth:', unsloth.__version__)
from unsloth import FastLanguageModel; print('FastLanguageModel OK')
from unsloth import is_bfloat16_supported; print('bfloat16 support:', is_bfloat16_supported())
import torch; print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', torch.cuda.get_device_properties(0).total_mem / 1024**3, 'GB')
