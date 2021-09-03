import pandas as pd
import numpy as np
import random
import sklearn
import math
import random
import string
import os
random.seed(0)
import json


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

    def __init__(self, wi, wo, no):

        self.ni = len(wi)
        self.nh = len(wo)
        self.no = no


        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no


        self.wi = wi
        self.wo = wo

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
            #self.ai[i] = sigmoid(inputs[i])
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

    def test(self, patterns):
        result = self.update(patterns)[0]
        print(result)
        result = (((result+1.0)/2.0)*0.2 + 0.4 - 0.5)*100

        if result> 0:
            score = int(math.ceil(result))
        else:
            score = int(math.floor(result))
        return score

    def output(self, patterns):
        result = self.update(patterns)[0]
        print(result)
        result = (((result+1.0)/2.0)*0.2 + 0.4 - 0.5)*100

        if result> 0:
            score = int(math.ceil(result))
        else:
            score = int(math.floor(result))
        return score

    def weights(self):
        print('inputweight:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('outputweight:')
        for j in range(self.nh):
            print(self.wo[j])



# def write_to_parquet(data, filename):
#     table = pa.Table.from_pandas(data)
#     pq.write_table(table, filename)

def demo():
    with open('./data/winRate2.json', encoding='utf-8') as f:
        winRate = json.load(f)

    wi = [
        [-2.364101368120024, 3.329238392573143]
        [-4.034727882776492, 2.6265674617670256]
        [-6.969363941873821, 4.149177598924891]
        [-3.3267768136916365, 2.7797458287259373]
        [-4.804430137272787, 4.065881620569871]
        [-7.655862360114982, 10.801980873489525]
    ]


    wo = [
        [-1.0509774062737813]
        [0.6317157933662332]
    ]


    n = NN(wi, wo, 1)

    # example: jax,leesin,syndra,sivir,sett
    ########## garen,gragas,yasuo,missfortune,blitzcrank
    input1 = input('please input first team champion name (separated by commas,as top,ju,mid,bot,sup):\n')
    input2 = input('please input second team champion name (separated by commas,as top,ju,mid,bot,sup):\n')
    n1 = input1.strip().split(',')
    print(n1)
    n2 = input2.strip().split(',')
    print(n2)
    x = []
    for t1,t2 in zip(n1, n2):
        # print()
        winrate1 = winRate.get(t1, -1)
        # print(winrate1)
        if winrate1 == -1:
            print("error")
        winrate = winrate1.get(t2, -1)
        if winrate==-1:
            print("error")
        x.append(winrate-0.5)

    print(x)
    score = n.output(x)
    print('Score prediction： %d' % (score))


demo()

# testData = [
#     [[1,2,3,4,5],[0]]
#     [[6,7,8,9,10],[0]]
# ]
