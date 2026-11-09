cmake_minimum_required(VERSION 3.20)
project(mace_libmace_aoti VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LIBMACE_SOURCE_DIR "" CACHE PATH "libmace source at the pinned revision")
set(libmace_revision e3413ae658530654894b6bac6a62a764a74c5115)
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LIBMACE_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT revision STREQUAL libmace_revision)
  message(FATAL_ERROR "libmace source must be at ${libmace_revision}")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${LIBMACE_SOURCE_DIR}" diff --exit-code HEAD --
  src/neighbour_list.cpp src/neighbour_list.h src/arrays.h src/fortran/mace.f90
  COMMAND_ERROR_IS_FATAL ANY)

# Python is needed only to locate the exporting installation at build time.
# The library links ATen/AOTInductor and never links libtorch_python.
include("${CMAKE_CURRENT_SOURCE_DIR}/../native/torch_runtime.cmake")
add_library(mace SHARED mace.cpp c_api.cpp "${LIBMACE_SOURCE_DIR}/src/neighbour_list.cpp")
target_include_directories(mace PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:include>" PRIVATE "${LIBMACE_SOURCE_DIR}/src")
mace_link_torch(mace PUBLIC)
set_target_properties(mace PROPERTIES INSTALL_RPATH "${torch_root}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE)
add_library(mace::mace ALIAS mace)
install(TARGETS mace EXPORT mace-targets LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES mace.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

option(MACE_WITH_FORTRAN "Build the compatible upstream MPI Fortran module" ON)
if(MACE_WITH_FORTRAN)
  enable_language(Fortran)
  find_package(MPI REQUIRED COMPONENTS Fortran)
  # Two corrections to the pinned module, without editing the source checkout:
  # non-root ranks own no C pointer, and usleep already takes microseconds.
  file(READ "${LIBMACE_SOURCE_DIR}/src/fortran/mace.f90" fortran_source)
  string(REPLACE "type(c_ptr) :: ptr !" "type(c_ptr) :: ptr = c_null_ptr !"
    fortran_source "${fortran_source}")
  string(REPLACE "dummy = usleep(iternval_usec * 1000)"
    "if (.not. completed) dummy = usleep(iternval_usec)"
    fortran_source "${fortran_source}")
  file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/mace.f90"
    CONTENT "! Derived from libmace ${libmace_revision}; see CMakeLists.txt for the two corrections.\n${fortran_source}" @ONLY)
  add_library(mace_fortran STATIC "${CMAKE_CURRENT_BINARY_DIR}/mace.f90")
  set_target_properties(mace_fortran PROPERTIES EXPORT_NAME fortran
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")
  target_include_directories(mace_fortran PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/modules>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_link_libraries(mace_fortran PUBLIC mace MPI::MPI_Fortran)
  add_library(mace::fortran ALIAS mace_fortran)
  install(TARGETS mace_fortran EXPORT mace-targets ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/mace.f90" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

configure_package_config_file(mace-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/mace-config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mace")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/mace-config-version.cmake"
  COMPATIBILITY SameMajorVersion)
install(EXPORT mace-targets NAMESPACE mace:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mace")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/mace-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/mace-config-version.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mace")

set(MACE_TEST_DRIVER_DIR "" CACHE PATH "Optional tests/integrations/libmace directory")
if(MACE_TEST_DRIVER_DIR)
  add_executable(mace_libmace_driver "${MACE_TEST_DRIVER_DIR}/driver.cpp")
  target_link_libraries(mace_libmace_driver PRIVATE mace)
  target_compile_features(mace_libmace_driver PRIVATE cxx_std_17)
  if(MACE_WITH_FORTRAN)
    add_executable(mace_libmace_fortran "${MACE_TEST_DRIVER_DIR}/driver.f90")
    target_link_libraries(mace_libmace_fortran PRIVATE mace_fortran)
  endif()
endif()
