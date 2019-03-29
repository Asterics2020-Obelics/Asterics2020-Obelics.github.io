'''
	Auteur : Pierre Aubert
	Mail : aubertp7@gmail.com
	Licence : CeCILL-C
'''

from setuptools import setup
from setuptools import Extension
from setuptools import find_packages
import sys
import os
from platform import system
import subprocess

import numpy as np

try:
	from Cython.Distutils import build_ext
except ImportError:
	use_cython = False
	print("Cython not found")
	raise Exception('Please install Cython on your system')
else:
	use_cython = True

NAME = 'sgemmpython'
VERSION = '1.0.0'
AUTHOR = 'Asterics developers'
AUTHOR_EMAIL = 'pierre.aubert@lapp.in2p3.fr'
URL = ''
DESCRIPTION = 'Asterics HPC sgemm python module'
LICENSE = 'Cecil-C'

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
	if type(value) == str:
		value = value.replace("-Wstrict-prototypes", "")
		value = value.replace("-DNDEBUG", "")
		cfg_vars[key] = value

extra_compile_args = ['-Werror', '-march=native',  '-mtune=native', '-ftree-vectorize', '-mavx2', '-O3', '-DVECTOR_ALIGNEMENT=32', '-g']

packageName = 'sgemmpython'
ext_modules = [
	Extension(packageName, ['@CMAKE_CURRENT_SOURCE_DIR@/sgemmpython.cpp',
				'@CMAKE_CURRENT_SOURCE_DIR@/../sgemm_intrinsics_pitch.cpp',
				'@CMAKE_CURRENT_SOURCE_DIR@/sgemmWrapper.cpp'
	],
	libraries=[],
	library_dirs=[],
	runtime_library_dirs=[],
	extra_link_args=[],
	extra_compile_args=extra_compile_args,

	include_dirs=['@CMAKE_CURRENT_SOURCE_DIR@/',
			'@CMAKE_CURRENT_SOURCE_DIR@/../',
			np.get_include()]
	)
]

try:
	setup(name = NAME,
		version=VERSION,
		ext_modules=ext_modules,
		description=DESCRIPTION,
		install_requires=['numpy', 'cython'],
		author=AUTHOR,
		author_email=AUTHOR_EMAIL,
		license=LICENSE,
		url=URL,
		classifiers=[
			'Intended Audience :: Science/Research',
			'License :: OSI Approved ::Cecil-C',
			'Operating System :: OS Independent',
			'Programming Language :: Python :: 3',
			'Topic :: Scientific/Engineering :: Astronomy',
			'Development Status :: 3 - Alpha'],
	)
except Exception as e:
	print(str(e))
	sys.exit(-1)

