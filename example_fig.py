import os
import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

sys.path.insert(1, os.path.join(os.path.expanduser("~"), "project", 'profiling'))  # for loading profile pickles

matplotlib.rcParams["figure.figsize"] = (8, 2)  # (4, 1.3)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

profile_pickle_filename_format = "./data/profile_pickles_a6000/{model}_{bs}_profile.pickle"

# read utilization pickle for all models
models = ["resnet50", "vgg13", "bert-base", "gpt2-medium"]
model_plotting_kwargs = {
    # "bert-base": ("tab:blue", "^"),
    # "resnet50": ("tab:orange", "o"),
}
    
# batch_sizes = list(utilization_pickle[models[0]].keys())
batch_sizes = [1, 2, 4, 8, 16]
print(f"batch_sizes {batch_sizes}")

fig, axes = plt.subplots(1, 4)

for i, model in enumerate(models):
    ax = axes[i]
    ax.tick_params(axis='both', which='major', labelsize=12)
    latencies = []
    throughputs = []
    for bs in batch_sizes:
        filename = profile_pickle_filename_format.format(model=model, bs=bs)
        with open(filename, "rb") as f:
            profile = pickle.load(f)
            latencies.append(profile.fwd_latency)
            throughputs.append(1000 / profile.fwd_latency * bs)  # num requests per second
    
    # plt.plot(latencies, utilizations, label=model, 
    ax.plot(latencies, throughputs, label=model, 
             color="tab:blue", 
             marker="o")  # "#1f77b4"

    y_value_diff = max(throughputs) - min(throughputs)
    ax.set_ylim(
        min(throughputs) - 0.1 * y_value_diff,
        max(throughputs) + 0.1 * y_value_diff,
    )
    x_value_diff = max(latencies) - min(latencies)
    ax.set_xlim(
        min(latencies) - 0.1 * x_value_diff,
        max(latencies) + 0.1 * x_value_diff,
    )
    if i == 0:
        ax.set_yticks([0, 250, 500])
    elif i == 1:
        ax.set_xticks([10, 15, 20])
    elif i == 3:
        ax.set_xticks([0, 500, 1000])

    # ax.set_yticks([100, 300, 500])
    # ax.set_ylabel(f"Utilization (%)", fontsize=12)
    # ax.set_ylabel(f"QPS", fontsize=12)
    ax.set_xlabel(f"Latency (ms)", fontsize=14)
    # ax.legend(loc="lower right")
    ax.set_title(model, fontsize=14)
    # ax.legend(loc="upper right")
    # ax.yaxis.set_label_coords(-0.15, 0.3)
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

axes[0].set_ylabel(f"Throughput (qps)", fontsize=14)
axes[0].yaxis.set_label_coords(-0.34, 0.35)
plt.tight_layout()
# fig.savefig(f'utilization_latency_tradeoff.png', bbox_inches='tight', dpi=500)
fig.savefig(f'throughput_latency_tradeoff.pdf', bbox_inches='tight', dpi=500)
plt.close()