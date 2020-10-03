import pandas as pd
import os
import seaborn as sn
import matplotlib.pyplot as plt

DIR = "clustersData"

def combineFiles():
    df = pd.DataFrame()

    print("Iterating over files...")
    for file in os.listdir(DIR):
        if file.endswith(".csv") and 'all_clusters' not in file:
            path = os.path.join(DIR, file)
            df = df.append(pd.read_csv(path), ignore_index=True) 
    df.to_csv(os.path.join(DIR, 'all_clusters.csv'))

    # drop the label
    df = df.drop(labels='label', axis=1)

    # create and plot the correlation matrix
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()

if __name__ == "__main__":
    combineFiles()