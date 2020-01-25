import os

import seaborn as sns

results_dir = "/Volumes/Macintosh HD/Users/khurramjaved96/Results_ICML/Plot"
results_dict = {}
std_dict = {}
import pandas as pd
import json
import matplotlib.pyplot as plt

data = []
folders = []
exeriments_data = []
for experiment_name in os.listdir(results_dir):
    if "DS_St" not in experiment_name:
        for rank in os.listdir(os.path.join(results_dir, experiment_name)):
            if "DS_St" not in rank:
                for seed in os.listdir(os.path.join(os.path.join(results_dir, experiment_name), rank)):
                    if "DS_St" not in seed:
                        with open(os.path.join(results_dir, experiment_name, rank, seed, "metadata.json")) as json_file:
                            folders.append(experiment_name + rank + seed)
                            try:
                                data_temp = json.load(json_file)
                                exeriments_data.append(data_temp)
                                print(experiment_name)
                                data.append(data_temp['results']['Running adaption running_loss'][0:3000])
                            except:
                                pass
                    # print(data)e
    # quit()
    # print(f)

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=0.4, rc={"lines.linewidth": 2.0})

mem = '0'
counter = 0
for current_experimnet_data in data:
    dictionary_temp = {}
    dictionary_temp["x"] = list(range(len(current_experimnet_data)))
    dictionary_temp["y"] = current_experimnet_data
    # print(results_dict[folder])
    df = pd.DataFrame(dictionary_temp)
    print(df)
    # df['#Classes'] = df.index
    # print(df)
    # print(df.columns)
    # df = pd.DataFrame.from_dict(results_dict[folder], orient='index')
    # df['Episode'] = df.index
    # df = df.melt('#Classes', var_name='cols', value_name='Accuracy')
    # print(df)
    # # print(df['vals'])
    plt.ylim(0, 4)

    sns.lineplot(x='x', y='y', data=df, legend='full', label=folders[counter])
    counter += 1
    #

    #     # plt.errorbar(list(results_dict[folder].keys()), list(results_dict[folder].values()), yerr= list(std_dict[folder].values()) , marker='s')

plt.tight_layout()
plt.savefig("plots/rebuttal_train.pdf", format="pdf")
quit()
