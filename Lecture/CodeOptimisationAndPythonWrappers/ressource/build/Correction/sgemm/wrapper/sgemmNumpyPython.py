'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

import sys
import numpy as np
import astericshpc

def getTimeFunctionSize(nbRepetition, nbElement):
	matX = np.asarray(np.random.random((nbElement, nbElement)), dtype=np.float32)
	matY = np.asarray(np.random.random((nbElement, nbElement)), dtype=np.float32)
	matRes = np.zeros((nbElement,nbElement), dtype=np.float32)
	
	timeBegin = astericshpc.rdtsc()
	for i in range(0, nbRepetition):
		np.matmul(matX, matY, matRes)
	
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
	makeElapsedTimeValue(listSize, 1000000)

