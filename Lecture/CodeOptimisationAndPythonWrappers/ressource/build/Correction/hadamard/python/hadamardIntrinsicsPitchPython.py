'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

import sys
import astericshpc
import hadamardpython

def allocInitTable(nbElement):
	tab = astericshpc.allocTable(nbElement)
	for i in range(0, nbElement):
		tab[i] = float(i*32%17)
	return tab

def getTimeFunctionSize(nbRepetition, nbElement):
	tabX = allocInitTable(nbElement)
	tabY = allocInitTable(nbElement)
	tabRes = astericshpc.allocTable(nbElement)
	
	timeBegin = astericshpc.rdtsc()
	for i in range(0, nbRepetition):
		hadamardpython.hadamard(tabRes, tabX, tabY)
	
	timeEnd = astericshpc.rdtsc()
	elapsedTime = float(timeEnd - timeBegin)/float(nbRepetition)
	elapsedTimePerElement = elapsedTime/float(nbElement)
	print("nbElement =",nbElement,", elapsedTimePerElement =",elapsedTimePerElement,"cy/el",", elapsedTime =",elapsedTime,"cy")
	print(str(nbElement) + "\t" + str(elapsedTimePerElement) + "\t" + str(elapsedTime),file=sys.stderr)

def makeElapsedTimeValue(listSize, nbRepetition):
	for val in listSize:
		getTimeFunctionSize(nbRepetition, val)

if __name__ == "__main__":
	listSize = [1000,
			1600,
			2000,
			2400,
			2664,
			3000,
			4000,
			5000,
			10000]
	makeElapsedTimeValue(listSize, 10000000)

