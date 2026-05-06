# %%
from transformers import AutoTokenizer

# 1. 모델에 맞는 토크나이저 로드
# AutoTokenizer는 모델 이름만 주어지면 적절한 토크나이저 클래스를 자동으로 선택함.
# (예: bert-base-uncased의 경우 BertTokenizer를 로드)
# 로드된 데이터(tokenizer.json, vocab.txt 등)는 ~/.cache/huggingface에 캐싱됨.
# use_fast=True: Rust로 구현된 Fast Tokenizer를 명시적으로 사용하도록 설정
tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# Fast Tokenizer 사용 여부 확인
print(f"Using Fast Tokenizer: {tok.is_fast}")

# %%
# 2. 텍스트를 입력 데이터(Batch)로 변환
# - padding: 배치 내의 문장 길이를 가장 긴 문장에 맞춤
# - truncation: 모델의 최대 입력 길이를 초과하는 경우 자름
# - return_tensors: "pt"(PyTorch), "np"(NumPy), "tf"(TensorFlow) 중 선택
batch = tok(
    ["hello world", "this is a test"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# batch는 dict 형태이며 보통 다음 키를 포함함:
# - input_ids: 토큰의 정수 인덱스
# - attention_mask: 실제 토큰(1)과 패딩 토큰(0)을 구분
# - token_type_ids: (BERT 등에서) 문장 구분용 (A 문장인지 B 문장인지)
print(batch.keys())
print("Input IDs shape:", batch["input_ids"].shape)

# %%
# 3. 디코딩 (Token ID -> String)
# [tok.decode vs tok.batch_decode]
# - tok.decode(): 단일 시퀀스(1D 리스트/텐서)를 문자열 하나로 변환.
#   예: tok.decode(batch["input_ids"][0])
# - tok.batch_decode(): 시퀀스 목록(2D 리스트/텐서)을 문자열 리스트로 변환.
#   배치 단위 데이터를 처리할 때 적합함.

# skip_special_tokens=True 옵션을 사용하면 [CLS], [SEP], [PAD] 등을 제거하고 원문만 확인 가능
decoded = tok.batch_decode(batch["input_ids"], skip_special_tokens=False)
print("Decoded strings (with special tokens):", decoded)

decoded_clean = tok.batch_decode(batch["input_ids"], skip_special_tokens=True)
print("Decoded strings (clean):", decoded_clean)

# %%
save_dir = "./tmp_tok"

# tokenizer의 설정, special token 정보, vocabulary 관련 파일 등을 저장함.
tok.save_pretrained(save_dir)

# 저장된 디렉토리에서 tokenizer를 다시 로드함.
tok2 = AutoTokenizer.from_pretrained(save_dir)

print(f"{type(tok2)      = }")
print(f"{tok2.is_fast    = }")
print(f"{tok2.vocab_size = }")

# %%
