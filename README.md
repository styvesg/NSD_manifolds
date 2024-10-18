# Variation in the geometry of concept manifolds across human visual cortex

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13948507.svg)](https://doi.org/10.5281/zenodo.13948507)

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)

# Overview

The main purpose of this collection of notebooks is to show the analysis that resulted in Figs. 2-6 of the paper. 

# Repo contents

| File | Description |
|------|-------------|
|`data_preparation.ipynb`| Some preprocessing of the NSD images and object segmentation maps for convenience. |
|`gnet8j_predict_imagenet.ipynb`| NSD subjects predictions for the ImageNet images. |
|`gnet8j_imagenet_manifold_rc.ipynb`| Characterization of high-dimensional geometry for the predicted ImageNet concepts. |
|`data_supercategory_image_structure_rc.ipynb`| Characterization of high-dimensional geometry of concept manifold for the NSD data (Fig. 2).|
|`gnet8j_supercategory_manifold_rc.ipynb`| Characterization of the high-dimensional geometry of concept manifold for the GNet. |
|`network_supercategory_manifold_rc.ipynb`| Characterization of the high-dimensional geometry of concept manifold for internal representation of DNNs. |
|`brain_manifold_analysis_rc.ipynb`| Analysis of few-show accuracy and geometric SNR for all the conditions. Figs. 3-6. |


# Data requirements

Data: [http://naturalscenesdataset.org/](http://naturalscenesdataset.org/) / 
 [A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence](https://www.nature.com/articles/s41593-021-00962-x)

Model and training:  [https://github.com/styvesg/nsd_gnet8x](https://github.com/styvesg/nsd_gnet8x) / 
 [Brain-optimized neural networks learn non-hierarchical models of representation in human visual cortex](https://www.nature.com/articles/s41467-023-38674-4)
This repo includes the trained parameters for a GNet8j model (a GNet joint model of NSD's 8 subjects) over a larger set of voxels than those used in the paper above. This voxel population covers most of the ROI labelled voxel under the Kastner altas of NSD.

# System requirements
## Software Requirements

The exact minimal requirement are not known exactly. However, the implementation uses mostly standard routines and very few dependencies, therefore old versions may still perform adequately. The numerical latest experiments and analysis has been performed with the following software versions:

- python 3.6.8
- numpy 1.19.5
- scipy 1.5.4
- torch 1.10 with CUDA 11.3 and cudnn 8.2
