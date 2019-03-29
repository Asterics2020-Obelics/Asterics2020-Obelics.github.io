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

#include "reductionWrapper.h"

std::string reductionWrapper_docstring = "Compute a reduction with aligned table of float32\n\
Parameters :\n\
	tabX : table of value (float32 aligned)\n\
Return :\n\
	Sum of the elements of tabX";

static PyMethodDef _reduction_methods[] = {
	{"reduction", (PyCFunction)reductionWrapper, METH_VARARGS, reductionWrapper_docstring.c_str()},
	{NULL, NULL}
};

static PyModuleDef _reduction_module = {
	PyModuleDef_HEAD_INIT,
	"reductionpython",
	"",
	-1,
	_reduction_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

///Create the python module reduction
/**	@return python module reduction
*/
PyMODINIT_FUNC PyInit_reductionpython(void){
	PyObject *m;
	import_array();
	
	m = PyModule_Create(&_reduction_module);
	if(m == NULL){
		return NULL;
	}
	return m;
}

