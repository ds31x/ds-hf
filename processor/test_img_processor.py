# %%
from image_processing_simple_vision import (
    SimpleVisionImageProcessor,
)

# 1. 인스턴스 생성
proc = SimpleVisionImageProcessor(size=224)

# 2. save_pretrained()가
#    preprocessor_config.json을 자동 생성
proc.save_pretrained(
    "./simple_vision_proc",
)
# %%
from transformers import AutoImageProcessor

p = AutoImageProcessor.from_pretrained(
    "./simple_vision_proc",
    trust_remote_code=True,
)
print(type(p), p.size)
# <class '...SimpleVisionImageProcessor'> 224


# %%
