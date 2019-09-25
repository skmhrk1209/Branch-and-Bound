cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

project(pycvxqp)

find_package(Boost COMPONENTS mpi serialization REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(pybind11 REQUIRED)

set(OSQP_INCLUDE_DIR /home/osqp/include)
set(OSQP_EIGEN_INCLUDE_DIR /home/osqp-eigen/include)

include_directories( 
    ${Boost_INCLUDE_DIRS} 
    ${EIGEN3_INCLUDE_DIR} 
    ${OSQP_INCLUDE_DIR} 
    ${OSQP_EIGEN_INCLUDE_DIR}
)

add_library(
    cvxqp SHARED 
    cvxqp.cpp
)
target_link_libraries(
    cvxqp INTERFACE 
    ${Boost_LIBRARIES}
    OsqpEigen::OsqpEigen 
    osqp::osqp
)

pybind11_add_module(
    pycvxqp MODULE
    pycvxqp.cpp
)
target_link_libraries(
    pycvxqp PRIVATE 
    cvxqp
)

add_executable(
    test 
    test.cpp
)
target_link_libraries(
    test PRIVATE 
    cvxqp
)
