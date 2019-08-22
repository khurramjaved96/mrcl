import seaborn as sns
import csv
import os
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
results_dir = "/Volumes/Macintosh HD/Users/khurramjaved96/Beluga/Train"
results_dict = {}
std_dict = {}
import pandas as pd
import json
import matplotlib.pyplot as plt

data = []
folders = []
for f in os.listdir(results_dir):
    if "DS_St" not in f:
        folders.append(f)
        with open(os.path.join(results_dir, f, "metadata.json")) as json_file:
            data_temp = json.load(json_file)
            print(f)
            data.append(data_temp['results']['Final Results'])
        # print(data)e
    # quit()
    # print(f)

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=0.4 , rc={"lines.linewidth": 2.0})

mem = '0'
counter=0
for dd in data:
    temp_data = {}
    for a in dd:
        # print(a)
        if a[0] in temp_data:
            if mem in a[1]:
                temp_data[a[0]].append(a[1][mem][0])
            else:
                temp_data[a[0]].append(a[1]['100'][0])
            # print(temp_data)
        else:
            temp_data[a[0]] = []
    d = temp_data
    # print(results_dict[folder])
    df = pd.DataFrame(d).T
    df['#Classes'] = df.index
    # print(df)
    # print(df.columns)
    # df = pd.DataFrame.from_dict(results_dict[folder], orient='index')
    # df['Episode'] = df.index
    df = df.melt('#Classes', var_name='cols', value_name='Accuracy')
    # print(df)
    # # print(df['vals'])
    # plt.ylim(0, 1)
    print(folders[counter])
    sns.lineplot(x='#Classes', y='Accuracy', data=df, legend='full',label =folders[counter])
    counter+=1
    #
    #
    #     # plt.errorbar(list(results_dict[folder].keys()), list(results_dict[folder].values()), yerr= list(std_dict[folder].values()) , marker='s')

plt.tight_layout()
plt.savefig("plots/rebuttal_train.pdf", format="pdf")
quit()