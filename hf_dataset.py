# %%
from datasets import load_dataset

dd = load_dataset("imdb")
print(f"{type(dd) = }")
print(dd)

# %%
print(dd.keys())
print(type(dd["train"]))
print(dd["train"])

# %%
sample = dd["train"][0]
print(type(sample))
print(sample)

# %%
dd["train"].column_names

# %%
dd["train"].features

# %%
# Hugging Face Datasets 라이브러리를 사용하여
# "imdb" 데이터셋의 train 분할(split)만 로드함.
# 결과는 하나의 Dataset 객체로 반환됨.
train_only = load_dataset("imdb", split="train")

# 로드한 train 데이터셋을 다시 학습용(train)과 평가용(test)으로 분할함.
# test_size=0.2 는 전체 데이터의 20%를 test 쪽으로 분리하겠다는 뜻임.
# seed=42 는 난수 시드를 고정하여, 매번 같은 방식으로 분할되게 하기 위한 설정임.
# 결과는 DatasetDict 형태로 반환되며,
# 일반적으로 "train", "test" 두 개의 키를 가짐.
dd2 = train_only.train_test_split(test_size=0.2, seed=42)

# 분할된 전체 구조를 출력함.
# 각 split 이름과 각 split에 포함된 샘플 수, feature 정보 등을 확인할 수 있음.
print(dd2)

# DatasetDict의 key 목록을 출력함.
# 보통 dict_keys(['train', 'test']) 와 같이 표시됨.
print(dd2.keys())
# %%
from datasets import DatasetDict # DatasetDict 클래스를 import함.
# 여러 개의 데이터 분할(split)을
# 하나의 사전(dict)처럼 묶어 관리할 때 사용함.
dd3 = DatasetDict({
    # dd2에서 "train" split을 꺼내어
    # 새 DatasetDict의 "train" split으로 넣음.
    "train": dd2["train"],

    # dd2에서 "test" split을 꺼내어
    # 이름을 "validation"으로 바꾸어 넣음.
    # 즉, test 데이터를 validation 데이터처럼 재구성하는 것임.
    "validation": dd2["test"],
})

# dd3 전체 구조를 출력함.
# 각 split 이름, 샘플 수, feature 정보 등을 확인할 수 있음.
print(dd3)

# dd3에 들어 있는 split 이름들만 출력함.
# 보통 dict_keys(['train', 'validation']) 형태로 나타남.
print(dd3.keys())
# %%
# 아래의 방법으로 dd2의 test를 직접 validation으로 바꿀 수 있음.
# pop을 하면 제거도 같이 되므로 이후로는 test로 접근불가.
dd2["validataion"] = dd2.pop("test")
print(dd2.keys())

# -----------------------------------------
# 7.

# %%

# %%
