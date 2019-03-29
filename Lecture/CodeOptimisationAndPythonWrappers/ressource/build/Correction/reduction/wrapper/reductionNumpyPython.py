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
	
	timeBegin = astericshpc.rdtsc()
	for i in range(0, nbRepetition):
		res = tabX.sum()
	
	timeEnd = astericshpc.rdtsc()
	elapsedTime = float(timeEnd - timeBegin)/float(nbRepetition)
	elapsedTimePerElement = elapsedTime/float(nbElement)
	print("nbElement =",nbElement,", elapsedTimePerElement =",elapsedTimePerElement,"cy/el",", elapsedTime =",elapsedTime,"cy")
	print(str(nbElement) + "\t" + str(elapsedTimePerElement) + "\t" + str(elapsedTime),file=sys.stderr)

def makeElapsedTimeValue(listSize, nbRepetition):
	for val in listSize:
		getTimeFunctionSize(nbRepetition, val)

if __name__ == "__main__":
	listSize = [	1024,
			2048,
			3072,
			4992,
			10048]
	makeElapsedTimeValue(listSize, 1000000)

