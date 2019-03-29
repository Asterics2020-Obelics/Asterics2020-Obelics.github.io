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

#include "hadamard_intrinsics_pitch.h"
#include "hadamardWrapper.h"

///Do the hadamard computation
/**	@param self : parent of the function if it exist
 * 	@param args : arguments passed to the function
 * 	@return result of the Hadamard product
*/
PyObject * hadamardWrapper(PyObject *self, PyObject *args){
	PyArrayObject *objTabX = NULL, *objTabY = NULL, *objTabRes = NULL;
	
	if(!PyArg_ParseTuple(args, "OOO", &objTabRes, &objTabX, &objTabY)){
		PyErr_SetString(PyExc_RuntimeError, "hadamardWrapper : wrong set of arguments. Expect tabRes, tabX, tabY\n");
		return NULL;
	}
	if(PyArray_NDIM(objTabX) != 1 || PyArray_NDIM(objTabY) != 1 || PyArray_NDIM(objTabRes) != 1){
		PyErr_SetString(PyExc_TypeError, "hadamardWrapper : input table must be a one dimension array");
		return NULL;
	}
	if(PyArray_DIMS(objTabX)[0] != PyArray_DIMS(objTabY)[0] || PyArray_DIMS(objTabX)[0] != PyArray_DIMS(objTabRes)[0]){
		PyErr_SetString(PyExc_TypeError, "hadamardWrapper : input table must be of the same size");
		return NULL;
	}
	long unsigned int sizeElement(PyArray_DIMS(objTabX)[0]);
	
	const float * tabX = (const float*)PyArray_DATA(objTabX);
	const float * tabY = (const float*)PyArray_DATA(objTabY);
	float * tabRes = (float*)PyArray_DATA(objTabRes);
	
	hadamard_product(tabRes, tabX, tabY, sizeElement);
	
	Py_RETURN_NONE;
}

