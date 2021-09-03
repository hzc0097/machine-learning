import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

def pca(X, K):
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    eigVal, eigVec = np.linalg.eig(cov)
    eigVec = eigVec.T
    sort = np.argsort(eigVal)[::-1]
    eigVec = eigVec[sort]
    principles = eigVec[0:K]
    X = X - mean
    dimReduced = np.dot(X, principles.T)
    return dimReduced


def Sort(sub):
    sub.sort(key = lambda x: x[1], reverse=True)
    return sub


df = pd.read_parquet('fullDataPASFaster.parquet')
df.drop(['summonerName'], 1, inplace=True)
df.fillna(0, inplace=True)


X = np.array(df.drop(['win'], 1))
y = np.array(df['win'])
model = LogisticRegression(max_iter=100000)
df.drop(['win'], 1, inplace=True)
model.fit(X,y)
importance = model.coef_[0]
valueList = list(df.columns.values)
print(list(df.columns.values))


finalList = []
for i,v in enumerate(importance):
    finalList.append([valueList[i], abs(v)])
    print(f"Feature: {i} {valueList[i]}, Importance Score: {v}")
print(Sort(finalList))


plt.bar([x for x in range(len(importance))], importance)
plt.title("Most Important Features")
plt.xlabel("Feature")
plt.ylabel("Effect on win-rate")
plt.grid()
plt.show()