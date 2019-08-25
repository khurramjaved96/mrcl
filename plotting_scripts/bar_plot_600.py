import os

import seaborn as sns

results_dir = "/Volumes/Macintosh HD/Users/khurramjaved96/MRCL_talk/UncorrvsCorr"
results_dict = {}
std_dict = {}
import pandas as pd
import json
import matplotlib.pyplot as plt

data = {}
folders = []
for f in os.listdir(results_dir):
    if "DS_St" not in f:
        folders.append(f)
        with open(os.path.join(results_dir, f, "metadata.json")) as json_file:
            data_temp = json.load(json_file)
            print(f)
            # print([x[1]['0'][0] for x in data_temp['results']['Final Results']])
            data[f] = [x[1]['0'][0] for x in data_temp['results']['Final Results']]
        # print(data)e
    # quit()
    # print(f)

sns.set(style="whitegrid")
sns.set_context("paper", rc={"lines.linewidth": 2.0})

mem = '0'
counter = 0
df = pd.DataFrame(data)
print(df.describe())
# print(df.info)
print(df.head())
print(df.columns)

df = pd.melt(df)
ax = sns.barplot(x="variable", y="value", data=df)

# axes = ax.axes


# plt.tight_layout()
plt.ylim(0.82)
plt.savefig("barplots.pdf", format="pdf")
quit()
