import pyarrow.parquet as pq
import pandas as pd

try:
    table = pq.read_table(r'/Users/guoxing.lan/projects/datasets/math/gsm8k_for_ppo/train.parquet')
    print("File is readable.")
    print("Schema:", table.schema)
except Exception as e:
    print(f"Error reading file: {e}")

df1 = pd.read_parquet(r'/Users/guoxing.lan/projects/datasets/math/gsm8k_for_ppo/train.parquet')
dict1=df1.to_dict('records')

pass