import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("L1_buckets.csv")

plt.subplots(figsize=(10, 6))
plt.scatter(x = df['count'], y = df['L1'])
plt.xlabel("Count")
plt.ylabel("L1")

plt.savefig('L1_60_epochs.png')
