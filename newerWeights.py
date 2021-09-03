import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

def Sort(sub):
    sub.sort(key = lambda x: x[1], reverse=True)
    return sub

df = pd.read_parquet('biggerCombined.parquet')
df.drop(['summonerName'], 1, inplace=True)


df.fillna(0, inplace=True)

X = np.array(df.drop(['win'], 1))
X = StandardScaler().fit_transform(X)

y = np.array(df['win'])
model = LogisticRegression(max_iter=100000)

model.fit(X,y)
importance = model.coef_[0]
df = df.drop(['win'], 1)
valueList = list(df.columns.values)

finalList = []
weightList = []
for i,v in enumerate(importance):
    weightList.append(v)
    finalList.append([valueList[i], v])
    print(f"Feature: {i} {valueList[i]}, Importance Score: {v}")
print(finalList)
with open('weightOutputWATCH.txt', 'wb') as f:
    pickle.dump(weightList, f)
f.close()
finalList.sort(key = lambda i: i[1])
print(finalList)


plt.bar([x for x in range(len(importance))], importance)
plt.title("Logistic Regression of most important features")
plt.xlabel("Feature # (will add names later)")
plt.ylabel("Effect on win-rate")
plt.grid()
plt.show()
