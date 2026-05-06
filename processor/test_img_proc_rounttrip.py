# %%
from image_processing_simple_vision import (
    SimpleVisionImageProcessor,
)
from PIL import Image

# 인스턴스 생성
p = SimpleVisionImageProcessor(
    size=224,
    do_resize=True,
    do_normalize=True,
)

# 전처리 수행
img = Image.open("cat.jpg").convert("RGB")
out = p(img, return_tensors="pt")
print(out["pixel_values"].shape)
# torch.Size([1, 3, 224, 224])

# 저장
save_dir = "./tmp_custom_imgproc"
p.save_pretrained(save_dir)
# %%
p2 = SimpleVisionImageProcessor \
    .from_pretrained(save_dir)
out2 = p2(img, return_tensors="pt")

diff = (
    out["pixel_values"]
    - out2["pixel_values"]
).abs().max().item()

print(diff)  # 0.0
# %%
from transformers import AutoImageProcessor

p2 = AutoImageProcessor.from_pretrained(
    save_dir,
    trust_remote_code=True,
)
out2 = p2(img, return_tensors="pt")

diff = (
    out["pixel_values"]
    - out2["pixel_values"]
).abs().max().item()

print(diff)  # 0.0
# %%
