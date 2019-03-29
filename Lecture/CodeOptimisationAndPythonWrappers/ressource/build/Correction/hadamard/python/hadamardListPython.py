'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

import sys
import numpy as np
import astericshpc

def getTimeFunctionSize(nbRepetition, nbElement):
	tabX = np.asarray(np.random.random(nbElement), dtype=np.float32)
	tabY = np.asarray(np.random.random(nbElement), dtype=np.float32)
	tabRes = np.zeros(nbElement, dtype=np.float32)
	
	listX = []
	listY = []
	listRes = []
	for i in range(0,nbElement):
		listX.append(tabX[i])
		listY.append(tabY[i])
		listRes.append(tabRes[i]);
	
	timeBegin = astericshpc.rdtsc()
	for i in range(0, nbRepetition):
		for j in range(0, nbElement):
			listRes[j] = listX[j]*listY[j]
	
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
	makeElapsedTimeValue(listSize, 10000)

