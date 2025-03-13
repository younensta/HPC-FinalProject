
# HPC Final Project

This project aims to profile and optimize an algorithm that uses a spectral method to solve the famous Navier-Stokes equation.. The spectral method is based on the Fast Fourier Transform (FFT) to determine the coefficients of the basis functions. The details of the implementation can be found [here](https://levelup.gitconnected.com/create-your-own-navier-stokes-spectral-method-fluid-simulation-with-python-3f37405524f4).

## Versions

The code comes in four different versions.

* `original-code.py` is the most basic version, beeing the base code with structural optimizations, such as avoiding redundant calls.
* `In-place version.py is an implementation with in-place operations.`
* `cython-fft-navier.py is a version of the code where the FFT is compiled, to be executed you first need to execute the following commmand ``$python setup.py build_ext --inplace` to compile `.pyx` files
* `cupy-code.py` is a version to execute on a GPU, using `CuPy `library.

### Authors

* Xavier ROCHETTE
* Guillaume LE BRONNEC
* Youn AIRIAUD
