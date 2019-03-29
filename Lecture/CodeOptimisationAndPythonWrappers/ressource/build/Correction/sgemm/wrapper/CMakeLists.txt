project(SgemmWrapper)
cmake_minimum_required(VERSION 3.0)

file(GLOB moduleSrc "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

createPythonModule(install_sgemm_module setup.py ${moduleSrc})

runPythonExample(sgemmIntrinsicsPitchPython.py install_sgemm_module)
runPythonExample(sgemmNumpyPython.py install_asterics_hpc_module)

plotPerf("sgemmBasePy" sgemmNumpyPython sgemm_swap_Ofast sgemm_vectorize_Ofast)

plotPerf("sgemmSummaryPython" sgemmNumpyPython sgemm_swap_Ofast sgemm_vectorize_Ofast sgemmIntrinsicsPitchPython sgemm_intrinsicsPitch_O3)

