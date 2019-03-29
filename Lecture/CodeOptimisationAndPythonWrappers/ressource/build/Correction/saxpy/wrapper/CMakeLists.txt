project(SaxpyWrapper)
cmake_minimum_required(VERSION 3.0)

file(GLOB moduleSrc "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

createPythonModule(install_saxpy_module setup.py ${moduleSrc})

runPythonExample(saxpyIntrinsicsPython.py install_saxpy_module)
runPythonExample(saxpyBasePython.py install_asterics_hpc_module)
runPythonExample(saxpyNumpyPython.py install_asterics_hpc_module)

plotPerf("saxpyBasePy" saxpyBasePython saxpyNumpyPython saxpy_O3 saxpy_vectorize)

plotPerf("saxpySummaryPython" saxpy_O3 saxpyIntrinsicsPython saxpy_vectorize saxpy_intrinsics)

