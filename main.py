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

def PrepareRates(rates):
# convert rates 
	return [(rates[k+1]-rates[k]) / rates[k] for k in xrange(len(rates)-1)]

from pybrain.tools.shortcuts import buildNetwork
def BuildNetwork(INPUT_KERNEL):
	return buildNetwork(TRAIN_KERNEL, 15, 5, 1)

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
def TrainNetwork(network, data, INPUT_KERNEL, steps, epochs, continueEpochs):
	ds = SupervisedDataSet(TRAIN_KERNEL, 1)
	for i in  xrange(TRAIN_STEPS):
		ds.addSample(tuple(data[i:i+TRAIN_KERNEL]), (data[i + TRAIN_KERNEL]))
	trainer = BackpropTrainer(net, ds)
	trainer.trainUntilConvergence(maxEpochs = epochs, continueEpochs = continueEpochs)

def Predict(net, data, INPUT_KERNEL, num):
	P = []
	for i in xrange(num):
		indata = data[i:i+INPUT_KERNEL]
		P.append( net.activate(indata)[0] )
	return P

import math 
import matplotlib.pyplot as plt
import numpy as np
def PlotResults(truth, prediction):
	num = len(prediction)
	xlabels = np.linspace(0, len(prediction), len(prediction))

	T = truth
	P = prediction
	f, (plt1, plt2, plt3) = plt.subplots(3)
	plt1.plot(xlabels, T, 'r', xlabels, P, 'b')
	plt1.axhline(0)

	D = [math.fabs(T[i]-P[i]) for i in xrange(len(P))]
	S = [math.copysign(1., P[i]) * math.copysign(1., T[i]) for i in xrange(len(P))]

	plt2.plot(xlabels, D, 'g')
	plt2.axhline(0)

	plt3.plot(xlabels, S, 'o')
	plt3.margins(0.2)
	plt3.text(0.95, 0.05, 'correct trend precition: %s %%' % str(S.count(1)/float(len(S))) )

	plt.show()

##################
TRAIN_KERNEL = 20
TRAIN_STEPS = 500

data = PrepareRates(parseEuroXRef('trainingdata/eurofxref-hist.csv'))
net = BuildNetwork(TRAIN_KERNEL)
TrainNetwork(net, data, TRAIN_KERNEL, TRAIN_STEPS, 5, 5)

# testing

TEST_OFFSET = 2000
TEST_NUM = 100

P = Predict(net, data[TEST_OFFSET:], TRAIN_KERNEL, TEST_NUM)
T = data[TEST_OFFSET:TEST_OFFSET + TEST_NUM]

PlotResults(T, P)