'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

import sys
import astericshpc
import sgemmpython

def allocInitMatrix(nbElement):
	mat = astericshpc.allocMatrix(nbElement, nbElement)
	for i in range(0, nbElement):
		for j in range(0, nbElement):
			mat[i][j] = float((i*nbElement + j)*32%17)
	return mat

def getTimeFunctionSize(nbRepetition, nbElement):
	tabX = allocInitMatrix(nbElement)
	tabY = allocInitMatrix(nbElement)
	tabRes = astericshpc.allocMatrix(nbElement, nbElement)
	
	timeBegin = astericshpc.rdtsc()
	for i in range(0, nbRepetition):
		sgemmpython.sgemm(tabRes, tabX, tabY)
	
	timeEnd = astericshpc.rdtsc()
	elapsedTime = float(timeEnd - timeBegin)/float(nbRepetition)
	elapsedTimePerElement = elapsedTime/float(nbElement*nbElement)
	print("nbElement =",nbElement,", elapsedTimePerElement =",elapsedTimePerElement,"cy/el",", elapsedTime =",elapsedTime,"cy")
	print(str(nbElement) + "\t" + str(elapsedTimePerElement) + "\t" + str(elapsedTime),file=sys.stderr)

def makeElapsedTimeValue(listSize, nbRepetition):
	for val in listSize:
		getTimeFunctionSize(nbRepetition, val)

if __name__ == "__main__":
	listSize = [	10,
			16,
			24,
			32,
			40,
			56,
			80,
			90,
			104]
	makeElapsedTimeValue(listSize, 100000)

