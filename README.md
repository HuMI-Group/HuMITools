#Humitools: Segment biological bodies

Whether you work with lower or upper leg human or mice muscle, segmenting hu

##Basic Install

###Basic install for windows: 
Download the folder “Windows” in the “executables” folder and double click the .exe file in the folder. 
###Basic install for macOS (Intel):
To be released
###Basic install for macOS (Apple silicon)
To be released

##Advanced Install 

This project is implemented in python. If you want to adapt the code, develop your own networks, or alter our preprocessing pipeline, you will need to set up a python environment. For the purposes of this guide, we will assume that you have conda installed on your pc and use an IDE that you are familiar with.

First: Clone the github repository to a folder
Second: Open the main folder in a terminal (this should contain a file called “humitools_environment.yml”)
In the terminal, type: “conda env create -f environment.yml”
Within your chosen IDE, set the newly created conda environment “Humitools” as the python interpreter. To test whether everything worked, navigate to the main_GUI.py file and run it. A GUI as depicted in the images below should open.


 

##Data requirements

Data for training/predict must be available either as a 3D .nifti or as a nifti.gz file. During the course of training/prediction, the data will be converted to 3D numpy, which will be stored in separate folders called “”image_numpy” for the images and “label_numpy” for the labels. These folders can be found in the defined output folder, in a separate folder called “TEMP”. Additionally, 4 .nifti files will be stored for quality control.
For training, both scans and labels should be in the same folder and share the same name. The manually segmented files should be marked with “-label” as a suffix to the filename. If you intend to perform automatic segmentation for body parts with bilateral symmetry, check the box “Split (left/right)”. The images will be mirrored for training and prediction. The number of labels to be entered on the left hand side is the number of bodies to be segmented (e.g.; count of muscles) +1 for the background class. If you provide a training dataset, the background should be labeled as “0”. 

Describe the output of the model
We cannot provide data
We cannot give help for your own stuff
describe all settings in the gui
Describe the exemplary dataset



If you have limited amounts of manually segmented nifti files (we do not recommend training, when you have less than 40 manually segmented files), we recommend the following workflow:
Train a network on all labeled nifties from your database for at least 200 epochs
Use the weights from training to predict the images you have not yet manually segmented
Use a tool like 3D Slicer (https://www.slicer.org/) to correct the predicted labels
Train the network again on roughly 4/5th of the labeled nifties. 
Use the weights from this training to predict the remaining fifth
Use a tool like 3D Slicer (https://www.slicer.org/) to correct the predicted labels
Repeat this process until you are satisfied with the results
Use the weights from the trained network on newly acquired data



##Network recommendations

All networks presented here are adaptations of published papers, often altered so as to fit into limited vRAM or CPU memory. The program will check for cuda availability and will run on the GPU if a Nvidia GPU with cuda support is found.
If possible, we recommend “Resunet” if you have no reason . If you have sufficient memory, increase batch size, so that it is divisible by 8. If you have no access to a computer that can train Resunet, use “unet”. 
“Hoffentlich_Alex” is a homebrew project and should not be used for projects that aim for publication. Our group has found that “Hoffentlich_Alex” beats “ResUnet” for the segmentation of calves and thighs in human and mice, but this model did not go through peer review in any shape or form. 
There will be no support for problems that arise from using “Hoffentlich_Alex”!

Describe everything visisble to the user
Demo exemplary data

What can and cannot be done

Write a "EASY HOW TO"
