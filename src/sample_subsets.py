import pandas as pd
import numpy as np


filenames_path = "../filenames.txt"

df = pd.read_csv(filenames_path)
val_idx = np.random.choice(277524, replace=False, size=27752)
train, validate, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.8*len(df))])

train.to_csv("train.txt")
validate.to_csv("val.txt")
test.to_csv("test.txt")