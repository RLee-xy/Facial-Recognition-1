# Facial Recognition with Different Feature Extractions
Final Project for EN.601.675 Machine Learning (Fall 2020)

Project Mentor: Darius Irani

Team Members: Ruixin Li (rli57), Kejia Ren (kren6), Longji Yin (lyin10), Zhongyuan Zheng (zzheng34)

## Description

Our project is about investigating different feature extraction methods (e.g. PCA, LDA, ICA, LBP, HoG) on human face recognition task, verifying their advances and better understanding their properties by experiment with several classifiers (e.g. k-NN, SVM, Deep net).

## Brief Instruction

To run and reproduce the results of our mini-project, please navigate to the write-up ["demo.ipynb"](demo.ipynb), which includes all necessary commands for applying our methods and showing visualizations.

Before that, please install the required external packages:

* PyTorch (version >= 1.7.0)
* scikit-learn == 0.23.2
* scikit-image == 0.16.2

Then please download the **cropped version** of [Extended Yale Face Database B](http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip) and extract the data under "/data" directory.

## Utilities

Below is a brief introduction of each folder and script:

* models: This folder saves all the trained models (feature extractors and classifiers).
* classifiers.py: This script includes our customized methods for training and saving classifier models.
* dataset.py: This is an implementation of ExtendedYaleFace dataset used to load and pre-process data.
* features.py: This is script includes methods used to apply different feature extractors.
* utils.py: This script includes utility codes for save/load models and plot visualizations.

## Acknowledgement

* [The Extended Yale Face Database B](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html)

 
