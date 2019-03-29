project(HadamardWrapper)
cmake_minimum_required(VERSION 3.0)

file(GLOB moduleSrc "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

createPythonModule(install_hadamard_module setup.py ${moduleSrc})

runPythonExample(hadamardIntrinsicsPitchPython.py install_hadamard_module)
runPythonExample(hadamardBasePython.py install_asterics_hpc_module)
runPythonExample(hadamardNumpyPython.py install_asterics_hpc_module)
runPythonExample(hadamardListPython.py install_asterics_hpc_module)

plotPerf("hadamardBasePy" hadamard_product_O3 hadamard_product_vectorize hadamardBasePython hadamardNumpyPython)

plotPerf("hadamardSummaryPython" hadamard_product_O3 hadamard_product_vectorize hadamard_product_intrinsics hadamardIntrinsicsPitchPython)

plotPerf("hadamardListPy" hadamard_product_O3 hadamardBasePython hadamardNumpyPython hadamardListPython)

