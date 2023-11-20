import pandas as pd

try:
    df = pd.read_csv("./../DataMeta/MAMe_dataset.csv")
except:
    df = pd.read_csv("./DataMeta/MAMe_dataset.csv")
#%%

df.groupby("Medium").count()["Museum"] / df.shape[0] * 100