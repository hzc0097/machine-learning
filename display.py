from matplotlib import pyplot as plt
import pickle
plt.style.use('dark_background')

fileList = ['ResultDT.txt', 'ResultMNB.txt', 'ResultKNN.txt', 'ResultNeural.txt', 'ResultSVM.txt']
xValueList = [100, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000,
                  40000, 80000, 100000, 150000, 200000, 300000, 400000]
fig, ax = plt.subplots(2,3)
j = 0
k = 0
for fileName in fileList:

    with open(fileName, 'rb') as f:
        log = pickle.load(f)
    f.close()


    fitTimes = []
    scoreTimes = []
    accuracies = []
    title = fileName[:-4]
    for i in range(len(log[1])):
        fitTimes.append(log[2][i])
        scoreTimes.append(log[3][i] - log[2][i])
        accuracies.append(log[1][i])
    ax[j,k].plot(xValueList, fitTimes, label=(fileName[6:-4] + " Fit"))
    ax[j,k].plot(xValueList, scoreTimes, label=(fileName[6:-4] + " Score"))
    ax[1,2].plot(xValueList, accuracies, label=fileName[6:-4])
    print(fileList)
    print(accuracies[-1])
    ax[j,k].set_title(fileName[6:-4])
    ax[j,k].set_ylabel("Time(s)")
    ax[j, k].set_xlabel("Data Size")
    print(f"{fileName} Fits: {fitTimes}")
    print(f"{fileName} Scores: {scoreTimes}")
    print(f"{fileName} Accuracies: {accuracies}")
    print(f"{fileName} Final Accuracy: {accuracies[-1]}")

    k += 1
    if k == 3:
        k = 0
        j = 1
labels = ["Fit", "Score"]
ax[1,2].legend(fontsize=8)
ax[1,2].set_xscale('log')
ax[1,2].set_title("Performance")
ax[1,2].set_ylabel("Accuracy")
ax[1,2].set_xlabel("Data Size")
fig.legend(labels = labels, fontsize=12)
fig.set_figwidth(13)
fig.set_figheight(8)





plt.show()




