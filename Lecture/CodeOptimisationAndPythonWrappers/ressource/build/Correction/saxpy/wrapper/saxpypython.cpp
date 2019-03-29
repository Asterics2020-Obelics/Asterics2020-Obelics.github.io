/***************************************
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
****************************************/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifndef DISABLE_COOL_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL core_ARRAY_API
#endif

#include <iostream>

#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>

#include "saxpyWrapper.h"

std::string saxpyWrapper_docstring = "Compute a Saxpy with aligned table of float32\n\
Parameters :\n\
	tabRes : table of the results (float32 aligned)\n\
	scal : scalar (float32)\n\
	tabX : table of value (float32 aligned)\n\
	tabY : table of value (float32 aligned)\n\
Return :\n\
	Result of the Saxpy as numpy array (tabRes)";

static PyMethodDef _saxpy_methods[] = {
	{"saxpy", (PyCFunction)saxpyWrapper, METH_VARARGS, saxpyWrapper_docstring.c_str()},
	{NULL, NULL}
};

static PyModuleDef _saxpy_module = {
	PyModuleDef_HEAD_INIT,
	"saxpypython",
	"",
	-1,
	_saxpy_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

///Create the python module saxpy
/**	@return python module saxpy
*/
PyMODINIT_FUNC PyInit_saxpypython(void){
	PyObject *m;
	import_array();
	
	m = PyModule_Create(&_saxpy_module);
	if(m == NULL){
		return NULL;
	}
	return m;
}

