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

#include "sgemmWrapper.h"

std::string sgemmWrapper_docstring = "Compute a SGEMM product with aligned matrices of float32 with a pitch\n\
Parameters :\n\
	matRes : matrix of the results (float32 aligned)\n\
	matX :   matrix of value (float32 aligned)\n\
	matY :   matrix of value (float32 aligned)";

static PyMethodDef _sgemm_methods[] = {
	{"sgemm", (PyCFunction)sgemmWrapper, METH_VARARGS, sgemmWrapper_docstring.c_str()},
	{NULL, NULL}
};

static PyModuleDef _sgemm_module = {
	PyModuleDef_HEAD_INIT,
	"sgemmpython",
	"",
	-1,
	_sgemm_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

///Create the python module sgemm
/**	@return python module sgemm
*/
PyMODINIT_FUNC PyInit_sgemmpython(void){
	PyObject *m;
	import_array();
	
	m = PyModule_Create(&_sgemm_module);
	if(m == NULL){
		return NULL;
	}
	return m;
}

