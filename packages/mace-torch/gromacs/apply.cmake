# Apply only the reviewed integration patch to its pinned upstream source.
cmake_minimum_required(VERSION 3.20)
find_package(Git REQUIRED)
if(NOT IS_DIRECTORY "${GROMACS_SOURCE_DIR}/.git")
  message(FATAL_ERROR "Set GROMACS_SOURCE_DIR to a clean clone of GROMACS 2026.3")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${GROMACS_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT revision STREQUAL "121090014570a53a17ea391bcddae45e5ea05eb4")
  message(FATAL_ERROR "GROMACS must be pinned to 121090014570a53a17ea391bcddae45e5ea05eb4")
endif()
set(patch "${CMAKE_CURRENT_LIST_DIR}/gromacs-2026.3.patch")
file(READ "${patch}" expected)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${GROMACS_SOURCE_DIR}" diff --unified=0 HEAD
  OUTPUT_VARIABLE changes COMMAND_ERROR_IS_FATAL ANY)
if(changes STREQUAL expected)
  message(STATUS "MACE native integration already applied")
elseif(changes STREQUAL "")
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${GROMACS_SOURCE_DIR}" apply --unidiff-zero --check "${patch}"
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${GROMACS_SOURCE_DIR}" apply --unidiff-zero "${patch}"
    COMMAND_ERROR_IS_FATAL ANY)
else()
  message(FATAL_ERROR "GROMACS checkout has other edits; use a clean pinned clone. Edits were preserved.")
endif()
