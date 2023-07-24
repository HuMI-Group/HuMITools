from settings import Settings

settings = Settings()
#####strings
name_json = 'humitools_settings.json'

str_model = 'Model'
str_losses = 'Losses'
str_output = 'Output'
str_train = 'Train'
str_predict = 'Predict'
str_split = 'Split (e.g. left/right leg)'
str_loadweights = 'Continue Training'
str_epochs = 'Epochs'
str_totallabels = 'Total Labels'
str_spatialres = 'Spatial Resolution'
str_batchsize = 'Batch Size'
str_learningrate = 'Learning Rate'


dropdown_stuff = {
    str_model: ['resunet', 'Basic_Network', 'unet', 'Hoffentlich_Alex', 'Hoffentlich_Alex_with_Modules', 'denseunet',
              'resunetwithmultiresblocks'],
    str_losses: ['Categorical Cross Entropy','Focal Tversky',
               'Focal Loss', 'Tversky', 'Unified Focal Loss']
}

path_stuff = {
    str_output: "Output Folder: ",
    str_train: "Training Folder:",
    str_predict: "Prediction Folder:"}

tooltiptext = {
        str_model : 'Choose the appropriate model for your set up',
        str_losses : 'If you are unsure choose: Unified Focal Loss',
        str_output : 'Choose the folder where the model and settings are supposed to be stored',
        str_train : 'Choose the folder with the training data',
        str_predict : 'Choose the folder with unlabeled Niftis',
        str_split : 'Split and mirror the given Niftis e.g. for left/right leg',
        str_loadweights : 'If a model is given in the output folder, use the saved weights for training',
        str_epochs : 'Number of epochs the model is to be trained, as a rule of thumb this should be in the range of 200-500',
        str_totallabels : 'Total number of Labels in your training/predict dataset',
        str_spatialres : 'Changes the internal resolution on which the network is trained, only change this if you are sure',
        str_batchsize : 'Batch Size for training, increase this number for better results, if you have enough memory',
        str_learningrate : 'Learning Rate, if you see bad results with the training, altering this number can improve results',
        'str_predictbutton': "Predict",
        'str_trainbutton': "Train",
}

checkbox = {
    str_split: settings.split_legs,
    str_loadweights: settings.loadWeigth
}

text_options = {
    str_epochs: settings.epochs,
    str_totallabels: settings.number_of_labels,
}

advances_options = {
    str_spatialres: settings.inputShape_create,
    str_batchsize: settings.batch_size,
    str_learningrate: settings.learning_rate,
}

