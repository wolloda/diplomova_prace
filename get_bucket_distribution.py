DIR_PATH = "/storage/brno6/home/tslaninakova/learned-indexes/MTree100k/"

import pandas as pd
from LMI import LMI

iterations = [1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200]

for i in range(20):
    for max_iter in iterations:
        li = LMI(DIR_PATH)
        df = li.get_dataset(normalize = False)

        training_specs = {"GMM": [{'comp': df["L1"].max(), 'cov_type': 'diag', 'max_iter': max_iter}, {'comp': df["L2"].max(), 'cov_type': 'diag', 'max_iter': max_iter}]} 

        df_result = li.train(df, training_specs, should_erase = True)

        df_result_grouped_L2 = df_result.groupby(['L2_pred'])[['L2']].count()
        df_result_grouped_L1 = df_result.groupby(['L1_pred'])[['L1']].count()
        df_result_grouped_L1_L2 = df_result.groupby(['L1_pred', 'L2_pred'])[['L2']].count()

        with pd.option_context('display.max_rows', int(df["L1"].max()) * int(df["L2"].max())):
            df_result_grouped_L2.to_csv(f'/storage/brno6/home/wollf/learned-indexes/learned-indexes/buckets_per_epoch_count/{max_iter}_epochs/L2_buckets_{i}.csv')
            df_result_grouped_L1.to_csv(f'/storage/brno6/home/wollf/learned-indexes/learned-indexes/buckets_per_epoch_count/{max_iter}_epochs/L1_buckets_{i}.csv')
            df_result_grouped_L1_L2.to_csv(f'/storage/brno6/home/wollf/learned-indexes/learned-indexes/buckets_per_epoch_count/{max_iter}_epochs/L1-L2_buckets_{i}.csv')
