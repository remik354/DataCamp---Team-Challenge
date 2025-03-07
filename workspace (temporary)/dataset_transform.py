import pandas as pd

df = pd.read_csv("./data/student_original.csv", sep=";")
print(df.columns)

df_sampled = df.sample(frac=0.2, random_state=42)
train_df = df.drop(df_sampled.index)

train_df.to_csv("./data/train.csv", index=False)
df_sampled.to_csv("./data/test.csv", index=False)
