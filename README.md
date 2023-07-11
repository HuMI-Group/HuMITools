# IMPORTANT: This README is preliminary and still under development!
# HuMITools: Segment biological bodies

Whether you work with muscles or brains of mice or men, manual segmentation of MR-images is needed often. This time consuming process can be accelerate by using artificial intelligence.
HuMITools is an OpenSource project that provides pretrained models or enable users to quickly train a model to their own personal needs and predict biological bodies.

## Installation

There are currently two options for installation. 
- Install Executable: Downloadable via the releases (https://github.com/HuMI-Group/HuMITools/releases).
- Advanced Installation: Follow the advanced installation guide (see below) to run the code in a python environment.

### Advanced Installation

This project is implemented in python. If you want to adapt the code, develop your own networks or alter our preprocessing pipeline, you will need to set up a python environment. For the purposes of this guide, we will assume that you have conda installed on your pc and use an IDE that you are familiar with.

    1. Clone the github repository to a folder
    2. Open the folder 'pyapp' in a terminal, this should contain a file called humitools_environment.yml”
    3. In the terminal, type: “conda env create -f “humitools_environment.yml”
    4. Within your chosen IDE open the project and set the newly created conda environment “HuMITools” as the python interpreter. 
    5. To test whether everything worked, navigate to the main_GUI.py file and run it. A GUI as depicted in the images below should open.

# User guide
HuMITools is a python tool to train segmentation networks and predict segmentations

## First use
Opening the GUI, it will ask for an output folder. The output folder is the place where models will be saved during training and labels after prediction. It is also a folder were all temporary details are stored during processing and where the current seession is saved as a .json. The .json is a file created during training and again saved after closing the GUI. It will save your latest settings so that you can easily continue your training or prediction another time.

## Tool layout
1.	Yellow: Defined output folder, which includes temp folder, settings (.json) and trained model (.pt).
2.	Red: Settings for training.
3.	Green: Data folder for training containing images and labels.
4.	Blue: Settings and data folder, containing images, for prediciton.

![A screenshot of the app](./assets/GUI_screenshot.png)

## Data requirements
1.	HuMITools accepts only (3D) nifty files (.nii or .nii.gz)
2.	Labels belonging to images should have the same name with an additional "-label" (i.e. dataset1.nii.gz and dataset1-label.nii.gz) and be stored in the same folder.
3.	The background of all label.nii.gz should labeled zero and will be automatically added as label in the code.
4.	The ROI labels should be integers and consistent over all datasets.

## Training
To train a network on your individual dataset, you need images that are already segmented. During training, images will be normalized, the background will be cut and all images will be zoomed into the same dimension. The workflow and an axplanation of settings is listed below:

1. Click on "What should be trained?" to choose a folder with images and their respective labels.
2. On the left side you can choose the model and a loss. For recommendations  for models read further down. The preprocessing will be done automatically.
3. You can train and existing model by checking the box "load-old-weights".
4. Input the total number of your labels, which is the number of bodies to be segmented (e.g. count of muscles per leg).
5. If you intend to perform automatic segmentation for body parts with bilateral symmetry (like legs) you can check the box “Split (left/right)”.
6. Choose the number of epochs the network should train on. To just look if everything is working, we recommend to train for 5 epochs. Later on, the best results will be achieved with 300-500 epochs.
7. Last but not least, you can add a note to remind yourself what you are doing with this run. The note and all other settings will be saved in the .json.
8. Click train.

Depending on your hadware the training might take a while. Before training, the data will be preprocessed. For that, all images and labels will be converted to 3D numpy, which will be stored in separate folders called "image_numpy" for the images and "label_numpy" for the labels. These folders can be found in the defined output folder, in a separate folder called "TEMP". Additionally, four .nii files will be stored that can be used to check if after proprocessing the labels are still in their correct places. The model (.pt) will be saved in the outputfolder after a few epochs and updated every few epochs.

### Advanced settings
Depending on your hardware the training might take very long. This can be improved by reducing the batch size or the resolution of the images.
On the other hand, if the training is fast you can improve the outcome by increasing the batch size or image resolution.

## Predicting
To predict images, you need a trained model. Either you train it yourself or, in case your data is similar to ours (out phase reconstruction from Dixon data), you can use pretrained models. The model (.pt) has to be in the output folder and by clicking on "What do you want to predict?" you can choose the folder with images that you want to create labels for. HuMITools only accepts niftis (.nii or .nii.gz). Two options have to be filled correctly for prediction to work: First, did you split the images when training (e.g. when training leg data)? Then that option has to be checked for predicting as well. Also, the total number of labels has to be the same as it was at the training. With those settings set, you can predict your images by clicking the button "Predict". This should be relatively fast, but depending on the amount of images and your hadware. The resulting labels will be saved in the output folder.

## Pretrained models
The pretrained models are trained on either upper or lower legs of human MRI scans.
We used a large, heterogenouse datasets as published in the paper below. 

[![DOI_heterogen_Paper](https://img.shields.io/badge/DOI-10.3390/diagnostics11101747-blue.svg)](https://www.mdpi.com/2075-4418/11/10/1747)

When using our models you have to check the split checkbox
Number of labels for upper legs: 8
Number of labels for lower legs: 7

# Recommended workflow
If you have limited amounts of manually segmented nifti files, we recommend the following workflow:

    1. Train a network on all images with labels from your database for at least 200 epochs
    2. Use the so trained model to predict the images you have not segmented yet
    3. Use a tool like 3D Slicer (https://www.slicer.org/) to manually refine the predicted labels
    4. Train the network again on previously finished labels and the so manually refinded labels 
    5. Repeat to predict, manually refine and retrain the model until your model predicts the labels perfectly well so that you do not have to manually refine anymore

## Network recommendations
All networks presented here are adaptations of published papers, often altered so as to fit into limited vRAM or CPU memory. The program will check for cuda availability and will run on the GPU if a Nvidia GPU with cuda support is found.
If possible, we recommend “Resunet” . If you have sufficient memory, increase batch size, so that it is divisible by 8. If you have no access to a computer that can train Resunet, use “unet”. 
“Hoffentlich_Alex” is a homebrew project and should not be used for projects that aim for publication. Our group has found that “Hoffentlich_Alex” beats “ResUnet” for the segmentation of calves and thighs in human and mice, but this model did not go through peer review in any shape or form. 
There will be no support for problems that arise from using “Hoffentlich_Alex”!

# Limitations
Unfortunately we cannot provide you with trainings datasets, due to data privacy. But we provide you with MR images of our own leg muscles, so that you can see on what the pretrained models are trained on.
When changing any of code, apply advanced settings or use a not implemented model, this is on your own risk.
