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

from pybrain.tools.shortcuts import buildNetwork

# training

TRAIN_KERNEL = 10
TRAIN_STEPS = 100

net = buildNetwork(TRAIN_KERNEL, 15, 5, 1)

from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(TRAIN_KERNEL, 1)

for i in  xrange(TRAIN_STEPS):
	ds.addSample(tuple(rates[i:i+TRAIN_KERNEL]), (rates[i + TRAIN_KERNEL]))

from pybrain.supervised.trainers import BackpropTrainer

trainer = BackpropTrainer(net, ds)
for i in xrange(100):
	trainer.train()
#print trainer.trainUntilConvergence()

# testing
import math 

TEST_OFFSET = 2000
TEST_NUM = 10
for i in xrange(TEST_OFFSET, TEST_OFFSET + TEST_NUM):
	data = rates[i:i+TRAIN_KERNEL]
	truth = rates[i+TRAIN_KERNEL]
	prediction = net.activate(data)[0]
	diff = truth-prediction

	print truth, prediction, diff