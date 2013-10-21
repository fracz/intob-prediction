from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(2, 3, 1)

def find(l, e):
	return next(i for i in xrange(len(l)) if l[i] == e)

# zwraca kursy z kolejnych dni jako tuple wartosci(w dolarach, jenach?, funtach)
def parseEuroXRef(file):
	f = open(file, 'r')
	header = f.readline().split(',')

	rates = []

	usdIndex = find(header, 'USD')
	jpyIndex = find(header, 'JPY')
	gbpIndex = find(header, 'GBP')

	for line in f:
		split = line.split(',')
		r = float(split[usdIndex]), float(split[jpyIndex]), float(split[gbpIndex])
		rates.append(r)

	f.close()
	rates.reverse()
	return rates

rates = parseEuroXRef('trainingdata/eurofxref-hist.csv')
print rates