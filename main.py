def find(l, e):
	return next(i for i in xrange(len(l)) if l[i] == e)

# zwraca kursy z kolejnych dni
def parseEuroXRef(file):
	f = open(file, 'r')
	header = f.readline().split(',')

	rates = []

	usdIndex = find(header, 'USD')

	for line in f:
		split = line.split(',')
		r = float(split[usdIndex])
		rates.append(r)

	f.close()
	rates.reverse()
	return rates

rates = parseEuroXRef('trainingdata/eurofxref-hist.csv')

# convert rates 
ratesRel = [(rates[k+1]-rates[k]) / rates[k] for k in xrange(len(rates)-1)]
rates = ratesRel

from pybrain.tools.shortcuts import buildNetwork

# training

TRAIN_KERNEL = 20
TRAIN_STEPS = 500

net = buildNetwork(TRAIN_KERNEL, 15, 5, 1)

from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(TRAIN_KERNEL, 1)

for i in  xrange(TRAIN_STEPS):
	vals = rates[i:i+TRAIN_KERNEL+1]
	ds.addSample(tuple(rates[i:i+TRAIN_KERNEL]), (rates[i + TRAIN_KERNEL]))

from pybrain.supervised.trainers import BackpropTrainer

trainer = BackpropTrainer(net, ds)
trainer.trainUntilConvergence(maxEpochs = 10, continueEpochs=10)

# testing
import math 

TEST_OFFSET = 2000
TEST_NUM = 100
P = []
for i in xrange(TEST_OFFSET, TEST_OFFSET + TEST_NUM):
	data = rates[i:i+TRAIN_KERNEL]
	P.append( net.activate(data)[0] )

import matplotlib.pyplot as plt
import numpy as np

x_labels = np.linspace(0, TEST_NUM, TEST_NUM)
T = rates[TEST_OFFSET:TEST_OFFSET + TEST_NUM]

f, (plt1, plt2, plt3) = plt.subplots(3)
plt1.plot(x_labels, T, 'r', x_labels, P, 'b')
plt1.axhline(0)

D = [math.fabs(T[i]-P[i]) for i in xrange(len(P))]
S = [math.copysign(1., P[i]) * math.copysign(1., T[i]) for i in xrange(len(P))]

print S.count(1)/float(len(S))

plt2.plot(x_labels, D, 'g')
plt2.axhline(0)

plt3.plot(x_labels, S, 'o')
plt3.margins(0.2)

plt.show()