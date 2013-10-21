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
print len(rates)

from pybrain.tools.shortcuts import buildNetwork

TRAIN_KERNEL = 10
TRAIN_STEPS = 20

net = buildNetwork(TRAIN_KERNEL, 3, 1)

from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(TRAIN_KERNEL, 1)
ds.addSample((0,0,0,0,0,0,0,0,0,0),(0))

for i in  xrange(TRAIN_STEPS):
	ds.addSample(tuple(rates[i:i+TRAIN_KERNEL]), (rates[i + TRAIN_KERNEL]))

from pybrain.supervised.trainers import BackpropTrainer

trainer = BackpropTrainer(net, ds)
trainer.train()
print trainer