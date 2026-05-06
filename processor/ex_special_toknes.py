# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "bert-base-uncased"

tok = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# 1. tokenizer에 새 special token 추가
num_added = tok.add_special_tokens({
    "additional_special_tokens": ["<image>", "<bbox>", "<ocr>"]
})

# 2. 새 token이 실제로 추가된 경우에만 embedding matrix 크기 조정
if num_added > 0:
    print(f"Before resizing: {model.get_input_embeddings().weight.shape[0]}")
    model.resize_token_embeddings(len(tok))
    print(f"After resizing: {model.get_input_embeddings().weight.shape[0]}")

# 3. 학습 전 초기 상태 저장
save_dir = "./model_with_added_tokens_init"
tok.save_pretrained(save_dir)
model.save_pretrained(save_dir)
# %%
tok = AutoTokenizer.from_pretrained("./model_with_added_tokens_init")
model = AutoModelForCausalLM.from_pretrained("./model_with_added_tokens_init")

print(f"Reloaded model embedding size: {model.get_input_embeddings().weight.shape[0]}")
print(f"Tokenizer vocab size: {len(tok)}")

