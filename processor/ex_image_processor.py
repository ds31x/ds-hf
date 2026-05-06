# %%
import requests
from transformers import AutoImageProcessor
from PIL import Image

# 1. 모델에 맞는 이미지 프로세서 로드
# AutoImageProcessor는 모델 이름에 따라 적절한 전처리 설정(Resize, Normalize 등)을 자동으로 로드함.
proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
print(f"Image Processor type: {type(proc)}")
print(f"{proc.is_fast = }")  # 이미지 프로세서에는 일반적으로 is_fast 속성이 없음

# 샘플 이미지 로드 (URL에서 스트리밍으로 읽어와 RGB로 변환)
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# img = Image.open("cat.jpg").convert("RGB")
print(f"Original image size: {img.size}")  # (width, height) 형태로 출력됨

# %%
# 2. 이미지를 모델 입력 데이터(Batch)로 변환
# - images: 이미지 객체 리스트 전달
# - return_tensors: "pt"(PyTorch) 텐서 형식으로 반환
batch = proc(
    images=[img], 
    return_tensors="pt",
    # return_tensors="np",
    )
# dict_keys(['pixel_values'])  (모델에 따라 추가 key가 있을 수 있음)

for k in batch.keys():
    print(f"{k}: {batch[k].shape}")
# pixel_values: (1, 3, 224, 224) 같은 shape의

print(f"{type(batch["pixel_values"]) = }")
print(batch["pixel_values"].shape)
# (1, 3, 224, 224) 같은 shape의 tensor객체임

# %%
save_dir = "./tmp_imgproc"
proc.save_pretrained(save_dir)

proc2 = AutoImageProcessor.from_pretrained(save_dir)
print(type(proc2))
# %%
