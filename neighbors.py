import pandas as pd
import numpy as np
import time
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle


def singleRunKNN():
    # read dataset
    df = pd.read_parquet('fullDataPASFaster.parquet')

    df.drop(['summonerName'], 1, inplace=True)
    df.fillna(0, inplace=True)

    X = np.array(df.drop(['win'], 1))
    y = np.array(df['win'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test, y_test)


    print(X)
    print(accuracy)

def assess(df, X_test, y_test):

    startTime = time.perf_counter()
    X_train = np.array(df.drop(['win'], 1))
    y_train = np.array(df['win'])
    clf = neighbors.KNeighborsClassifier()
    print(f"training start {time.perf_counter()}")
    clf.fit(X_train, y_train)
    print(f"training done {time.perf_counter()}")
    endTime = time.perf_counter()
    runTime = endTime - startTime

    accuracy = clf.score(X_test, y_test)
    print(f"testing done {time.perf_counter()}")
    endTime = time.perf_counter()
    runTime2 = endTime - startTime
    return clf, accuracy, runTime, runTime2


def kNNGraphRun():

    df = pd.read_parquet('fullDataPASFaster.parquet')

    df.drop(['summonerName'], 1, inplace=True)
    df.fillna(0, inplace=True)
    testdf = df[400000:]
    df = df[:400000]
    X_test = np.array(testdf.drop(['win'], 1))
    y_test = np.array(testdf['win'])

    testValues = [100, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 40000, 80000, 100000, 150000, 200000, 300000, 400000]
    accuracyList = []
    timeList = []
    time2List = []
    for i in testValues:
        tempdf = df[:i]
        model, accuracy, runTime, runTime2 = assess(tempdf, X_test, y_test)
        accuracyList.append(accuracy)
        timeList.append(runTime)
        time2List.append(runTime2)
        print(f"Finished size: {i}")

    fig, ax = plt.subplots(1,3)
    ax0 = ax[0]
    ax1 = ax[1]
    ax2 = ax[2]
    ax0.plot(testValues, accuracyList)
    ax1.plot(testValues, timeList)
    ax2.plot(testValues, time2List)

    ax2.set_title("Second RunTime against testing")

    ax0.set_yticks(np.arange(0,1.1,0.1))
    ax0.set_title("kNN Accuracy w.r.t. data size")
    ax0.set_xscale("log")
    ax0.set_xlabel("Training Data Size")
    ax0.set_ylabel("Accuracy Against Test Set")
    ax0.grid()
    ax1.set_title("RunTime")


    plt.show()

    print(testValues)
    print(accuracyList)
    print(timeList)
    print(time2List)
    resultDict = [testValues, accuracyList, timeList, time2List]
    with open('ResultKNN.txt', 'wb') as f:
        pickle.dump(resultDict,f)


if __name__=="__main__":
    singleRunKNN()
    kNNGraphRun()