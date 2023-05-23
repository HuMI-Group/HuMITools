from settings import Settings

settings = Settings()
dropdown_stuff = {
    "Model": ['resunet', 'Basic_Network', 'unet', 'Hoffentlich_Alex', 'Hoffentlich_Alex_with_Modules', 'denseunet',
              'resunetwithmultiresblocks'],
    "Losses": ['Focal Tversky', 'Categorical Cross Entropy',
               'Focal Loss', 'Tversky', 'Unified Focal Tversky']
}

path_stuff = {
    "Output": "Define an output folder for models, settings and temp-folder: ",
    "Train": "What should be trained (img+label)?",
    "Predict": "What do you want to predict (img)?"}

checkbox = {
    'Split (left/right leg)': settings.split_legs,
    'Load old Weights': settings.loadWeigth
}

text_options = {
    'Epochs': settings.epochs,
    'Total Labels': settings.number_of_labels,
    'Note': ''
}

advances_options = {
    'Spatial resolution': settings.inputShape_create,
    'Batch size': settings.batch_size,
    'Learning rate': settings.learning_rate,
}

name_json = 'humitools_settings_gui.json'
