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

#include "reduction_intrinsics_interleave8.h"

///Do the reduction computation
/**	@param self : parent of the function if it exist
 * 	@param args : arguments passed to the function
 * 	@return result of the reduction result
*/
PyObject * reductionWrapper(PyObject *self, PyObject *args){
	PyArrayObject *objTabX = NULL;
	if(!PyArg_ParseTuple(args, "O", &objTabX)){
		PyErr_SetString(PyExc_RuntimeError, "reductionWrapper : wrong set of arguments. Expect tabX\n");
		return NULL;
	}
	if(PyArray_NDIM(objTabX) != 1){
		PyErr_SetString(PyExc_TypeError, "reductionWrapper : input table must be a one dimension array");
		return NULL;
	}
	long unsigned int sizeElement(PyArray_DIMS(objTabX)[0]);
	
	const float * tabX = (const float*)PyArray_DATA(objTabX);
	float res(reduction(tabX, sizeElement));
	
	return Py_BuildValue("f", res);
}

