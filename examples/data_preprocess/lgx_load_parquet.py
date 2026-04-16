import pyarrow.parquet as pq
import pandas as pd
import json
# try:
#     table = pq.read_table(r'/Users/guoxing.lan/projects/datasets/math/gsm8k_for_ppo/train.parquet')
#     print("File is readable.")
#     print("Schema:", table.schema)
# except Exception as e:
#     print(f"Error reading file: {e}")

# 一种有效的对比方式
import pyarrow.parquet as pq
import json

path = "/Users/guoxing.lan/projects/datasets/math/gsm8k_for_ppo/train.parquet"
pf = pq.ParquetFile(path)
kv = {k.decode(): v.decode(errors="replace") for k, v in (pf.metadata.metadata or {}).items()}

hf = json.loads(kv["huggingface"])
print(hf["info"]["features"]["prompt"])

path = "/Users/guoxing.lan/projects/datasets/math/gsm8k_good/train.parquet"
pf = pq.ParquetFile(path)
kv = {k.decode(): v.decode(errors="replace") for k, v in (pf.metadata.metadata or {}).items()}

hf = json.loads(kv["huggingface"])
print(hf["info"]["features"]["prompt"])
# 一种有效的对比方式
pass


# 一种有效的对比方式
import pyarrow.parquet as pq
import json

def show_prompt_feature(path: str):
    pf = pq.ParquetFile(path)
    md = pf.metadata.metadata or {}
    kv = {k.decode(): v.decode(errors="replace") for k, v in md.items()}
    hf = json.loads(kv["huggingface"])
    print("===", path)
    print(json.dumps(hf["info"]["features"]["prompt"], ensure_ascii=False, indent=2))

show_prompt_feature("/Users/guoxing.lan/projects/datasets/math/gsm8k_for_ppo/train.parquet")
show_prompt_feature("/Users/guoxing.lan/projects/datasets/math/gsm8k_good/train.parquet")
# 一种有效的对比方式


df1 = pd.read_parquet(r'/Users/guoxing.lan/projects/datasets/math/gsm8k_for_ppo/train.parquet')

# df1 = pd.read_parquet(r'/Users/guoxing.lan/projects/datasets/math/gsm8k_good/train.parquet')
dict1=df1.to_dict('records')
print(dict1[0])
pass