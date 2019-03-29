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

#include "hadamardWrapper.h"

std::string hadamardWrapper_docstring = "Compute a Hadamard product with aligned table of float32\n\
Parameters :\n\
	tabRes : table of the results (float32 aligned)\n\
	tabX : table of value (float32 aligned)\n\
	tabY : table of value (float32 aligned)\n\
Return :\n\
	Result of the Hadamard as numpy array (tabRes)";

static PyMethodDef _hadamardpython_methods[] = {
	{"hadamard", (PyCFunction)hadamardWrapper, METH_VARARGS, hadamardWrapper_docstring.c_str()},

	{NULL, NULL}
};

static PyModuleDef _hadamardpython_module = {
	PyModuleDef_HEAD_INIT,
	"hadamardpython",
	"",
	-1,
	_hadamardpython_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

///Create the python module hadamardpython
/**	@return python module hadamardpython
*/
PyMODINIT_FUNC PyInit_hadamardpython(void){
	PyObject *m;
	import_array();
	
	m = PyModule_Create(&_hadamardpython_module);
	if(m == NULL){
		return NULL;
	}
	return m;
}

