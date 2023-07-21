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
        str_model : 'Model',
        str_losses : 'Losses',
        str_output : 'Output',
        str_train : 'Train',
        str_predict : 'Predict',
        str_split : 'Split (e.g. left/right leg)',
        str_loadweights : 'Continue Training',
        str_epochs : 'Epochs',
        str_totallabels : 'Total Labels',
        str_spatialres : 'Spatial Resolution',
        str_batchsize : 'Batch Size',
        str_learningrate : 'Learning Rate',
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

