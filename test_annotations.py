import gzip, json

path = "/home/andreina/Documents/Programs/Dataset/objaverse_assets/2023_07_28/annotations.json.gz"
with gzip.open(path) as f:
    ann = json.load(f)
sample_id = next(iter(ann))
print(list(ann[sample_id].keys()))