/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#define NO_IMPORT_ARRAY
#ifndef DISABLE_COOL_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL core_ARRAY_API
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <numpy/arrayobject.h>
#include <bytearrayobject.h>

#include "sgemm_intrinsics_pitch.h"
#include "sgemmWrapper.h"

///Get the pitch of a matrix
/**	@param nbCol : number of columns of the matrix
 * 	@return pitch of the matrix
*/
long unsigned int getPitch(long unsigned int nbCol){
	long unsigned int vecSize(VECTOR_ALIGNEMENT/sizeof(float));
	long unsigned int pitch(vecSize - (nbCol % vecSize));
	if(pitch == vecSize){pitch = 0lu;}
	return pitch;
}

///Do the hadamard computation
/**	@param self : parent of the function if it exist
 * 	@param args : arguments passed to the function
 * 	@return result of the Hadamard product
*/
PyObject * sgemmWrapper(PyObject *self, PyObject *args){
	PyArrayObject *objMatX = NULL, *objMatY = NULL, *objMatRes = NULL;
	
	if(!PyArg_ParseTuple(args, "OOO", &objMatRes, &objMatX, &objMatY)){
		PyErr_SetString(PyExc_RuntimeError, "sgemmWrapper : wrong set of arguments. Expect matRes, matX, matY\n");
		return NULL;
	}
	if(PyArray_NDIM(objMatX) != 2 || PyArray_NDIM(objMatY) != 2 || PyArray_NDIM(objMatRes) != 2){
		PyErr_SetString(PyExc_TypeError, "sgemmWrapper : input matrices must be a two dimension array");
		return NULL;
	}
	if(PyArray_DIMS(objMatX)[0] != PyArray_DIMS(objMatY)[0] || PyArray_DIMS(objMatX)[0] != PyArray_DIMS(objMatRes)[0] ||
		PyArray_DIMS(objMatX)[1] != PyArray_DIMS(objMatY)[1] || PyArray_DIMS(objMatX)[1] != PyArray_DIMS(objMatRes)[1] ||
		PyArray_DIMS(objMatX)[0] != PyArray_DIMS(objMatX)[1] ||
		PyArray_DIMS(objMatY)[0] != PyArray_DIMS(objMatY)[1] ||
		PyArray_DIMS(objMatRes)[0] != PyArray_DIMS(objMatRes)[1])
	{
		PyErr_SetString(PyExc_TypeError, "sgemmWrapper : input matrices must be of the same size and square");
		return NULL;
	}
	long unsigned int sizeElement(PyArray_DIMS(objMatX)[0]);
	
	const float * matX = (const float*)PyArray_DATA(objMatX);
	const float * matY = (const float*)PyArray_DATA(objMatY);
	float * matRes = (float*)PyArray_DATA(objMatRes);
	
	sgemm(matRes, matX, matY, sizeElement, getPitch(sizeElement));
	
	Py_RETURN_NONE;
}

