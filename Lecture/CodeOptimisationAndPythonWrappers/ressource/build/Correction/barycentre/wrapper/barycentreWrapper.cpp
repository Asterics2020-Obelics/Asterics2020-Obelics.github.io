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

#include "barycentre_intrinsics.h"
#include "barycentreWrapper.h"

///Do the barycentre computation
/**	@param self : parent of the function if it exist
 * 	@param args : arguments passed to the function
 * 	@return result of the barycentre
*/
PyObject * barycentreWrapper(PyObject *self, PyObject *args){
	PyArrayObject *objTabX = NULL, *objTabY = NULL, *objTabA = NULL;
	
	if(!PyArg_ParseTuple(args, "OOO", &objTabX, &objTabY, &objTabA)){
		PyErr_SetString(PyExc_RuntimeError, "barycentreWrapper : wrong set of arguments. Expect tabX, tabY, tabA\n");
		return NULL;
	}
	if(PyArray_NDIM(objTabX) != 1 || PyArray_NDIM(objTabY) != 1 || PyArray_NDIM(objTabA) != 1){
		PyErr_SetString(PyExc_TypeError, "barycentreWrapper : input table must be a one dimension array");
		return NULL;
	}
	if(PyArray_DIMS(objTabX)[0] != PyArray_DIMS(objTabY)[0] || PyArray_DIMS(objTabX)[0] != PyArray_DIMS(objTabA)[0]){
		PyErr_SetString(PyExc_TypeError, "barycentreWrapper : input table must be of the same size");
		return NULL;
	}
	long unsigned int sizeElement(PyArray_DIMS(objTabX)[0]);
	
	const float * tabX = (const float*)PyArray_DATA(objTabX);
	const float * tabY = (const float*)PyArray_DATA(objTabY);
	const float * tabA = (float*)PyArray_DATA(objTabA);
	float gx(0.0f), gy(0.0f);
	barycentre(gx, gy, tabX, tabY, tabA, sizeElement);
	
	return Py_BuildValue("ff", gx, gy);
}

