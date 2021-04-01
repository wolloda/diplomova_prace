import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

class Results:
    def __init__(self, file_name):
        self.file_name = file_name
        self.name = self.file_name[0:-4] # .csv

        self.to_replace = ["cov-type-", "max-iter-", "init-params-", "weight_concentration_prior_type-", "weight-concentration-prior-type-", "cov-type=", "max-iter=", "init-params=", "weight_concentration_prior_type=", "weight-concentration-prior-type="]

        for i in self.to_replace:
            self.name = self.name.replace(i, "")

        self.data = pd.read_csv(file_name)

def plot(file_locations, output_file):
    rows = len(file_locations)
    cols = 2
    space_size = 0.3
    fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize=(cols * 12, rows * 10 + (rows - 1) * space_size))
    fig.tight_layout()
    fig.subplots_adjust(hspace = space_size)
    break_points = [500, 1000, 3000, 5000, 10000, 50000, 100000, 200000, 300000, 500000, 750000]

    for row, model in enumerate(file_locations):
        plot_data(top_x(get_files(f"{model}/*.csv"), 8), row, axes, break_points )

    # plt.show()
    plt.savefig(output_file)

def plot_data(data, row, axes, breakpoints, col = -1):
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 1, len(data))]
    axes[row, 0].set_prop_cycle('color', colors)
    axes[row, 1].set_prop_cycle('color', colors)

    for result in data:
        if (col == -1 or col == 0):
            axes[row, 0].plot(breakpoints, result.data["model_recall"], marker = "x", label = result.name)
            axes[row, 0].set_xlabel("Number of objects in buckets visited")
            axes[row, 0].set_ylabel("Recall")
            axes[row, 0].grid()
            axes[row, 0].title.set_text("Recall / Number of objects in buckets visited")
            axes[row, 0].legend()
            axes[row, 0].grid(b = True)

        if (col == -1 or col == 1):
            axes[row, 1].plot(result.data["model_times"], result.data["model_recall"], marker = "x", label = result.name)
            axes[row, 1].set_xlabel("Time (seconds)")
            axes[row, 1].set_ylabel("Recall")
            axes[row, 1].title.set_text("Recall / Time (seconds)")
            axes[row, 1].legend()
            axes[row, 1].grid(b = True)

def get_files(csv_path):
    #get_files.counter += 1
    return [Results(file_name) for file_name in glob.glob(csv_path)]

# get_files.counter = 0

def top_x(data, x):
    # top_x.counter += 1
    return sorted(data, key = lambda x: x.data.iloc[-1]["model_times"])[0:x]
# top_x.counter = -1


plot(["COPHIR_1M/GMM/10-10-10-10-10/", "COPHIR_1M/GMM/10-10-10-10-10/ENCODED-100/", "COPHIR_1M/GMM/10-10-10-10-10/ENCODED-200/", "PROFI_1M/GMM/57-100/"], "performance_2.png")
