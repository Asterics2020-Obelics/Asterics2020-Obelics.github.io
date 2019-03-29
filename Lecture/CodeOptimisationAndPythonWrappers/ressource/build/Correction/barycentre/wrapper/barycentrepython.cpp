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

#include "barycentreWrapper.h"

std::string barycentreWrapper_docstring = "Compute a 2d barycentre with aligned table of float32\n\
Parameters :\n\
	tabX : table of value (float32 aligned)\n\
	tabY : table of value (float32 aligned)\n\
	tabA : table of value (float32 aligned)\n\
Return :\n\
	barycentre (x, y)";

static PyMethodDef _barycentre_methods[] = {
	{"barycentre", (PyCFunction)barycentreWrapper, METH_VARARGS, barycentreWrapper_docstring.c_str()},

	{NULL, NULL}
};

static PyModuleDef _barycentre_module = {
	PyModuleDef_HEAD_INIT,
	"barycentrepython",
	"",
	-1,
	_barycentre_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

///Create the python module barycentre
/**	@return python module barycentre
*/
PyMODINIT_FUNC PyInit_barycentrepython(void){
	PyObject *m;
	import_array();
	
	m = PyModule_Create(&_barycentre_module);
	if(m == NULL){
		return NULL;
	}
	return m;
}

