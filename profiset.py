from imports import *
def scale_per_descriptor_profiset(df, label_names):
    col_pos = 0
    normalized = []
    numerical = df.drop(label_names + ["object_id"], axis=1).values
    normalized.append(preprocessing.scale(numerical))
    df = df.drop(df.columns.difference(label_names+["object_id"]), 1)
    df = pd.concat([df, pd.DataFrame(np.hstack((normalized)))], axis=1)
    return df

def load_indexes_profiset(base_dir, label_names, filenames=['level-1.txt', 'level-2.txt'], should_deduplicate=True):
    df_i = pd.DataFrame([])
    filenames.reverse()
    for c, filename in enumerate(filenames):
        if c != 0: col_names = label_names[:-c]
        else: col_names = label_names
        df_ = pd.read_csv(base_dir+filename, names = col_names + ["object_id"], sep=r'[.+\s]', dtype=np.int64, engine='python', header=None)
        df_i = pd.concat([df_i, df_])
        if should_deduplicate:
            df_i = df_i.drop_duplicates(["object_id"])
        #print(df_i.shape)
    return df_i.apply(pd.to_numeric, errors='ignore')

def get_dataset_profiset(base_dir, label_names, filenames=['level-1.txt', 'level-2.txt']):
    index_df = load_indexes_profiset(base_dir, filenames)
    descr_df = pd.read_csv(f"{base_dir}/objects_descriptors.txt",  sep=r'[\s+]', engine='python', header=None)
    #df_orig = merge_dfs(numerical, labels, index_df)
    obj_ids = pd.read_csv(f"{base_dir}/objects_objectid.txt",  sep=r'[\s+]', engine='python', header=None)
    descr_df["object_id"] = obj_ids[2].values
    df_orig = pd.merge(descr_df, index_df, on=['object_id'], how = 'outer')
    df = scale_per_descriptor_profiset(df_orig)
    return df, df_orig

def get_1M_profiset(objects_path="/storage/brno6/home/tslaninakova/learned-indexes/datasets/descriptors-decaf-odd-5M-1.data", index_path=None, labels=None):
    #index_df = load_indexes_profiset(index_path, labels)
    #index_df = index_df.sort_values(by=["object_id"])
    df_odd = pd.read_csv(objects_path, header=None)
    arr_full = [np.fromstring(arr, dtype=np.float16, sep=" ") for arr in df_odd[0].values]
    #obj_ids = pd.read_csv(objects_path.replace("odd", "even"), sep=" ", names=["1", "2", "object_id"], header=None)[["object_id"]]
    #df_profi = pd.DataFrame(np.array(arr_full))
    #index_df = index_df.join([df_profi])
    return np.array(arr_full)

def get_profiset(objects_path="/storage/brno6/home/tslaninakova/learned-indexes/datasets/descriptors-decaf-odd-5M-1.data", 
                 indexes_path="/storage/brno6/home/tslaninakova/learned-indexes/MtreeProfi2000/"):
    index_df = load_indexes_profiset(indexes_path, ["L1", "L2"],  filenames=[f'level-{l}.txt' for l in range(1,3)])
    index_df = index_df.sort_values(by=["object_id"])
    assert index_df.shape[0] == 1000000
    data = pd.read_csv(objects_path, header=None, sep=" ", dtype=np.float16)
    data = data.drop(data.columns[-1], axis=1)
    data.reset_index(drop=True, inplace=True)
    index_df.reset_index(drop=True, inplace=True)
    df_full = pd.concat([data, index_df], axis=1)
    df_full = df_full.sample(frac=1)
    return df_full

