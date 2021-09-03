import pandas as pd

filename = 'biggerCombined.csv'
newFileName = 'biggerCombined.parquet'

df = pd.read_csv(filename)
df.to_parquet(newFileName)