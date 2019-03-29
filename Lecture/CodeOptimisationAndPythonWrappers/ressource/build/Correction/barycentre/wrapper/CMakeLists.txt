project(BarycentreWrapper)
cmake_minimum_required(VERSION 3.0)

file(GLOB moduleSrc "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

createPythonModule(install_barycentre_module setup.py ${moduleSrc})

runPythonExample(barycentreIntrinsicsPython.py install_barycentre_module)
runPythonExample(barycentreNumpyPython.py install_asterics_hpc_module)

plotPerf("barycentreBasePy" barycentreNumpyPython barycentre_base_O3)

plotPerf("barycentreSummaryPython" barycentreNumpyPython barycentre_base_O3 barycentreIntrinsicsPython barycentre_intrinsics)

