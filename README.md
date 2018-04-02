# A Parallelized Multi-Objective Dynamic Sampling Framework for Analytical Electron Microscopy

This is a work in progress supplement to the publication [Hujsak, Karl A., et al. "High Speed/Low Dose Analytical Electron Microscopy with Dynamic Sampling." Micron (2018)](https://www.sciencedirect.com/science/article/pii/S0968432817304821).  The goal of this project was to predict an optimal set of reduced points which could be interpolated to produce an accurate estimation elemental distributions in Scanning Transmission Electron Microscopes (STEMs).  As it is supervised technique, the user must supply training images with which the coefficients can be learned.  After producing many training pairs, essential parameters for interpolation and the heuristics are self estimated by running simulations on the training data.  As opposed to previous methods, this training is vectorized and parallelized for high speed.

The code in this project is currently WIP, and only supports training and simulations of single element maps.  In the next series of releases we will include the multi-objective framework as well as the Digital Micrograph plug-in to facilitate experiments.  Our framework can either run directly on the microscope computer, or communicate over the network to the DM system.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Currently tested with:

```
python 3.5.3
multiprocess 0.70.5
numpy 1.11.3
scikit-learn 0.18.1
scikit-image 0.12.3
h5py 2.6.0
```

## Running the scripts

```
python DynamicImageClass.py -n TrainingImage.tif
```

## Authors

* **Karl Hujsak** - *Initial work* - [khujsak](https://github.com/khujsak)


## Acknowledgments

* G. M. Dilshan Godaliyadda
