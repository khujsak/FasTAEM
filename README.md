# Dynamic-Sample v0.1

Scripts to execute a supervised learning approach for Dynamic Sampling.  Limited documentation currently available and code is still heavily in flux, your mileage may vary.  To be done:

1.  Flesh out Documentation
2.  Improve variable names and commonality
3.  Randomized Training Images
4.  Bayesian Linear Regression
5.  Posterior Predictive Variance Estimates
6.  Discrete Images and Classification Inpainting
7.  Hyperspectral Approaches

Currently only works in an environment like spyder.  Will not work currently executed in a command line.

To execute:

1.  Run distance matrix calculation.  This precomputes a large distance matrix for evaluating distances very quickly in the main code.  This creates a variable named 'disbig' which is important to have sitting in memory for future calculations.
2.  Run training script with your training images, which will store several .npy files for the Value and Feature vectors in your current directory.
3.  Train the coefficients, currently only available using least squares, eventually will include a Bayesian approach for estimating posterior predictive variance during an experiment.
4.  Simulate sequential imaging with make predictions.

No parameter estimates are automatically done, so play around with c and L to have a good fit.  Currently uses a Navier Stokes inpainting step from OpenCV, but eventually will also support a simple nearest neighbors approach.

Only supports continuous images, support for discrete and hyperspectral ongoing.

Requires:

Numpy, scipy, OpenCv 3, Numba (remove the @jit decorators in the dynsamp file if not), ect.
