import pandas as pd
import numpy as np
import random
import sklearn


def getData(filepath, test_size):
    file = open(filepath)
    x_train = []
    y_train = []
    gameId = []
    for line in file:
        a = line.strip().split(' ')
        x = []
        num = 1
        for i in range(len(a)):
            if i == 1:
                if a[i]=='1':
                    y_train.append([float(1)])
                if a[i]=='0':
                    y_train.append([float(-1)])
            elif i==0:
                gameId.append([int(a[i])])
            else:
                x.append((float(a[i])-0.5))
        x_train.append(x)
    dataSet = []
    for i in range(len(y_train)):
        l = []
        l.append(x_train[i]) # x
        l.append(y_train[i]) # y
        l.append(gameId[i]) # gameId
        dataSet.append(l)

    random.shuffle(dataSet)
    # dataSet = dataSet[]
    train_dataset = np.array(dataSet[:int(len(dataSet) * (1 - test_size))]) 
    test_dataset = np.array(dataSet[int(len(dataSet) * (1 - test_size)):]) 

    # x_train = train_dataset[:, :int(len(dataSet[0])-1)]
    # y_train = train_dataset[:, int(len(dataSet[0])-1)]
    #
    # x_test = test_dataset[:, :int(len(dataSet[0])-1)]
    # y_test = test_dataset[:, int(len(dataSet[0])-1)]
    file.close()
    # print(y_train)
    return dataSet, train_dataset, test_dataset


import math
import random
import string
import os
random.seed(0)


def rand(a, b):
    return (b-a)*random.random() + a


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m


def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y**2

class NN:

    def __init__(self, ni, nh, no):

        self.ni = ni + 1 
        self.nh = nh
        self.no = no


        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)


        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('error！')


        for i in range(self.ni-1):

            self.ai[i] = inputs[i]


        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)


        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N, M):

        if len(targets) != self.no:
            raise ValueError('error！')


        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error


        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error


        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print(N*change, M*self.co[j][k])


        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change


        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    def test(self, patterns):
        s = 0
        for p in patterns:
            result = self.update(p[0])[0]
            result = ((result+1.0)/2.0)*0.2 + 0.4 # [0.4 0.5]
            print(p[0], '->', result)
            label = p[1][0]
            if result>=0.5 and label==1:
                s += 1
            elif result<0.5 and label==-1:
                s += 1
        acc = float(s) / float(len(patterns))
        print('data: %d, true data:%d, acc, %-.5f' % (len(patterns), s, acc))

    def output(self, patterns):
        out = []
        for p in patterns:
            l = []
            result = self.update(p[0])[0]
            result = (((result+1.0)/2.0)*0.2 + 0.4 - 0.5)*100
            # print(result)
            if result> 0:
                score = int(math.ceil(result))
            else:
                score = int(math.floor(result))
            l.append(p[2][0])
            l.append(score)
            out.append(l)
            # print(p[0], '->', result)
        return out

    def weights(self):
        print('inputweight:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('outputweight:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.01, M=0.1):

        if os.path.exists('./data/loss.txt'):
            os.remove('./data/loss.txt')
        f = open('./data/loss.txt', 'a')
        for i in range(iterations):
            error = 0.0
            loss = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            loss = error / len(patterns)
            f.write(str(loss) + '\n')
            if i % 100 == 0:
                print('loss %-.5f' % loss)

        f.close()


# def write_to_parquet(data, filename):
#     table = pa.Table.from_pandas(data)
#     pq.write_table(table, filename)

def demo():

    dataSet, trainData, testData = getData('./data/match.txt', 0.3)

    trainData = trainData[:5000]
    testData = testData[:1000]
    # print(trainData[:2])
    # print(trainData)


    n = NN(5, 2, 1)

    n.train(trainData)

    n.test(testData)

    out = n.output(dataSet)
    if os.path.exists('./data/gameId_score.txt'):
        os.remove('./data/gameId_score.txt')
    with open('./data/gameId_score.txt', 'a') as f:
        for i in out:

            f.write(str(i[0]) + ' ' + str(i[1]) + '\n')


    n.weights()



demo()

# testData = [
#     [[1,2,3,4,5],[0]]
#     [[6,7,8,9,10],[0]]
# ]
