from mindex_new import MIndex
li = MIndex(PATH="/storage/brno6/home/tslaninakova/learned-indexes/", mindex="MIndex1M-Profiset-leaf2000/")
#from mindex import MIndex
import pickle
#from searching import *
from profiset import *

li.labels =  ["L1", "L2"]
index_df = load_indexes_profiset("/storage/brno6/home/tslaninakova/learned-indexes/MIndex1M-Profiset-leaf2000/", li.labels,  filenames=[f'level-{l}.txt' for l in range(1,3)])
index_df = index_df.sort_values(by=["object_id"])
assert index_df.shape[0] == 1000000
objects = get_1M_profiset(index_path="/storage/brno6/home/tslaninakova/learned-indexes/MIndex1M-Profiset-leaf2000/", objects_path="/storage/brno6/home/tslaninakova/learned-indexes/datasets/descriptors-decaf-odd-5M-1.data", labels=li.labels)
df = pd.DataFrame(objects)
print(df.shape)
df["L1"] = index_df["L1"].values
df["L2"] = index_df["L2"].values
df["object_id"] = index_df["object_id"].values
#li.labels = ["L1", "L2"]
#df, df_orig = li.get_dataset()
print("Loaded dataset")

param_dicts= [{"GMM": [{'comp': 50, 'cov_type': 'diag'}, {'comp': 100, 'cov_type': 'diag'}]}]
#param_dicts= [{"GMM": [{'comp': 200, 'cov_type': 'spherical'}, {'comp': 100, 'cov_type': 'spherical'}]}]

df_res = li.train_LMI(df, param_dicts[0])
print("Finished training")
