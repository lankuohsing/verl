from datasets import load_dataset
import datasets
ds = datasets.load_dataset("/Users/guoxing.lan/projects/datasets/math/gsm8k_parquet",cache_dir="/Users/guoxing.lan/projects/datasets/.cache")
ds.save_to_disk("/Users/guoxing.lan/projects/datasets/gsm8k")
ds1=datasets.load_from_disk("/Users/guoxing.lan/projects/datasets/gsm8k")

print(ds)
print(ds1)

pass
