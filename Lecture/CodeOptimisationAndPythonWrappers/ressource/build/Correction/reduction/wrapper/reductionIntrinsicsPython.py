'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

import sys
import astericshpc
import reductionpython

def allocInitTable(nbElement):
	tab = astericshpc.allocTable(nbElement)
	for i in range(0, nbElement):
		tab[i] = float(i*32%17)
	return tab

def getTimeHadamardSize(nbRepetition, nbElement):
	tabX = allocInitTable(nbElement)
	timeBegin = astericshpc.rdtsc()
	for i in range(0, nbRepetition):
		res = reductionpython.reduction(tabX)
	
	timeEnd = astericshpc.rdtsc()
	elapsedTime = float(timeEnd - timeBegin)/float(nbRepetition)
	elapsedTimePerElement = elapsedTime/float(nbElement)
	print("nbElement =",nbElement,", elapsedTimePerElement =",elapsedTimePerElement,"cy/el",", elapsedTime =",elapsedTime,"cy")
	print(str(nbElement) + "\t" + str(elapsedTimePerElement) + "\t" + str(elapsedTime),file=sys.stderr)

def makeElapsedTimeValue(listSize, nbRepetition):
	for val in listSize:
		getTimeHadamardSize(nbRepetition, val)

if __name__ == "__main__":
	listSize = [	1024,
			2048,
			3072,
			4992,
			10048]
	makeElapsedTimeValue(listSize, 10000000)

