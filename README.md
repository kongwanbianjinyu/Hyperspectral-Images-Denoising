# Hyperspectral Images Denoising via a Tensor Dictionary #


### Contents of folder ###
1. ourmethod.m
2. debug.m
3. nmodeproduct.m
4. TensorDictionaryLearning

* ourmethod.m
This file contains the code for our extension. The code can be run directly after adding the relevant paths (contained in TensorDictionaryLearning). 
To denoise a different image, either set msi equal to this image, or load a different .mat file. 
Change the value of sigma for different amplitudes of Gaussian noise. 
Mu, lambda, and params.spect_dim are adjustable parameters that yield differing results. The current values were found to give acceptable results while running in a reasonable amount of time. 
The code prints the value of the objective function during every outer loop iteration, once every 1000 inner loop iterations, and some quantitative performance measures at the end. 

* debug.m
This file contains the code used to debug the individual optimization functions. To use this, uncomment a block of code, marked by timestamps for when the test was passed. After running, the output is the value of the objective function for the optimal value of the variable in question. The value array contains values of the objective function when the variable takes some random values around the optimal value. It can be seen that all the values in this array exceed the objective function at the optimal point.

* nmodeproduct.m
This is a script we found online for computing the tensor product. It is used in ourmethod.m and debug.m.

* TensorDictionaryLearning
This folder contains files that are used in the comparison methods. The path program of tensor dictionary learning -> tdl_demo -> Comparison.m can be used to run the comparison methods at once. However, note that this will not immediately work, as parts of the code of the Tensor DL have been adapted to be used in ourmethod.m.
