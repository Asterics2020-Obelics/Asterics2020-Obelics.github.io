project(ReductionWrapper)
cmake_minimum_required(VERSION 3.0)

file(GLOB moduleSrc "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

createPythonModule(install_reduction_module setup.py ${moduleSrc})

runPythonExample(reductionIntrinsicsPython.py install_reduction_module)
runPythonExample(reductionNumpyPython.py install_asterics_hpc_module)

plotPerf("reductionBasePy" reductionNumpyPython reduction_real_Ofast reduction_real_vectorize_Ofast)

plotPerf("reductionSummaryPython" reductionNumpyPython reduction_real_Ofast reduction_real_vectorize_Ofast reductionIntrinsicsPython reduction_real_intrinsics_interleave8_O3)

