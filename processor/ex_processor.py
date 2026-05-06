# %%
from transformers import AutoProcessor
from PIL import Image
import requests

# 샘플 이미지 로드 (URL에서 스트리밍으로 읽어와 RGB로 변환)
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
print(f"Original image size: {img.size}")  # (width, height) 형태로 출력됨
img.save("cat.jpg")
# img = Image.open("cat.jpg").convert("RGB")

# %%
# 샘플 이미지 로드 (URL에서 스트리밍으로 읽어와 RGB로 변환)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
print(f"Original image size: {img.size}")  # (width, height) 형태로 출력됨
img.save("dog.jpg")
# img = Image.open("cat.jpg").convert("RGB")

# %%
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

out = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=[Image.open("cat.jpg").convert("RGB"), Image.open("dog.jpg").convert("RGB")],
    return_tensors="pt",
    padding=True,
)

for k in out.keys():
    print(f"{k}: {type(out[k]) = }{out[k].shape = }")
# input_ids, attention_mask, pixel_values ... 같은 형태

# %%
# 3. 프로세서 저장 및 로드
# processor는 tokenizer와 image_processor의 기능을 모두 포함함.
# 저장 시 설정 파일들과 각 구성 요소(tokenizer, image_processor 등)의 정보가 함께 저장됨.
save_dir = "./tmp_proc"
processor.save_pretrained(save_dir)

# 저장된 디렉토리에서 프로세서를 다시 로드함.
processor2 = AutoProcessor.from_pretrained(save_dir)

print(f"{type(processor2) = }")
# %%
