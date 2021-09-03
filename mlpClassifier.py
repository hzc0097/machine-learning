import pickle
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def assess(df, X_test, y_test):
    startTime = time.perf_counter()
    X_train = np.array(df.drop(['win'], 1))
    y_train = np.array(df['win'])
    print(f"training start {time.perf_counter()}")
    model = MLPClassifier(solver='lbfgs', random_state=1, verbose=False,
                          max_iter=10000, hidden_layer_sizes=(8,4),
                          learning_rate='adaptive', learning_rate_init=0.01).fit(X_train, y_train)
    model = MLPClassifier(max_iter=10000, verbose=True, hidden_layer_sizes=(16,8)).fit(X_train, y_train)
    endTime = time.perf_counter()
    runTime = endTime - startTime
    print(f"training done {time.perf_counter()}")
    accuracy = model.score(X_test, y_test)
    endTime = time.perf_counter()
    runTime2 = endTime - startTime
    print(f"testing done {time.perf_counter()}")
    return model, accuracy, runTime, runTime2

def MLPGraphRun():
    df = pd.read_parquet('biggerCombined.parquet')

    df.drop(['summonerName'], 1, inplace=True)
    df.fillna(0, inplace=True)

    scaler = preprocessing.MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])


    testdf = df[400000:]
    df = df[:400000]
    X_test = np.array(testdf.drop(['win'], 1))
    y_test = np.array(testdf['win'])

    testValues = [100, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000,
                  40000, 80000, 100000, 150000, 200000, 300000, 400000]
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
    fig, ax = plt.subplots(1, 3)
    ax0 = ax[0]
    ax1 = ax[1]
    ax2 = ax[2]
    ax0.plot(testValues, accuracyList)
    ax1.plot(testValues, timeList)
    ax2.plot(testValues, time2List)

    ax2.set_title("Train and Test Time")

    ax0.set_yticks(np.arange(0, 1.1, 0.1))
    ax0.set_title("MLP Accuracy w.r.t. data size")
    ax0.set_xscale("log")
    ax1.set_xscale("log")
    ax2.set_xscale("log")
    ax0.set_xlabel("Training Data Size")
    ax0.set_ylabel("Accuracy Against Test Set")
    ax0.grid()
    ax1.set_title("TrainTime")

    plt.show()

    print(testValues)
    print(accuracyList)
    print(timeList)
    print(time2List)
    resultDict = [testValues, accuracyList, timeList, time2List]
    return resultDict
    with open('ResultNN.txt', 'wb') as f:
        pickle.dump(resultDict, f)

def MLPSingleRun():
    df = pd.read_parquet('biggerCombined.parquet')
    print(len(df))
    df.drop(['summonerName'], 1, inplace=True)
    df.fillna(0, inplace=True)
    scaler = preprocessing.MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    X = np.array(df.drop(['win'], 1))
    y = np.array(df['win'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = MLPClassifier(max_iter=10000, verbose=True, hidden_layer_sizes=(32,16,8,2))
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy = {accuracy}")


if __name__ == "__main__":
    dict1 = MLPGraphRun()
    dict2 = MLPGraphRun()
    dict3 = MLPGraphRun()

    final = [dict1, dict2, dict3]
    result = [sum(e)/len(e) for e in zip(*final)]


    with open('ResultNN.txt', 'wb') as f:
        pickle.dump(result, f)
