'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

import sys
import astericshpc
import barycentrepython

def allocInitTable(nbElement):
	tab = astericshpc.allocTable(nbElement)
	for i in range(0, nbElement):
		tab[i] = float(i*32%17)
	return tab

def getTimeFunctionSize(nbRepetition, nbElement):
	tabX = allocInitTable(nbElement)
	tabY = allocInitTable(nbElement)
	tabA = allocInitTable(nbElement)
	
	timeBegin = astericshpc.rdtsc()
	for i in range(0, nbRepetition):
		g = barycentrepython.barycentre(tabX, tabY, tabA)
	
	timeEnd = astericshpc.rdtsc()
	elapsedTime = float(timeEnd - timeBegin)/float(nbRepetition)
	elapsedTimePerElement = elapsedTime/float(nbElement)
	print("nbElement =",nbElement,", elapsedTimePerElement =",elapsedTimePerElement,"cy/el",", elapsedTime =",elapsedTime,"cy")
	print(str(nbElement) + "\t" + str(elapsedTimePerElement) + "\t" + str(elapsedTime),file=sys.stderr)

def makeElapsedTimeValue(listSize, nbRepetition):
	for val in listSize:
		getTimeFunctionSize(nbRepetition, val)

if __name__ == "__main__":
	listSize = [1024,
			2000,
			2400,
			3024,
			5024,
			10000]
	makeElapsedTimeValue(listSize, 10000000)

