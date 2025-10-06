import pandas as pd
print("Checking English sampled data balance...")
df_en = pd.read_csv("data/raw/sampled_imdb_en.csv")
print(df_en['sentiment'].value_counts())

print("\n")
print("Checking Spanish sampled data balance...")

df_es = pd.read_csv("data/raw/sampled_imdb_es.csv")
print(df_es['sentimiento'].value_counts())