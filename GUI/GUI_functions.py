import os
import tkinter as tk
import traceback
import json
from tkinter import filedialog
from tkinter.constants import END
from tkinter.messagebox import showinfo, showerror
from torch import load
from torch import nn

from GUI import GUI_hardcoded
from humi_pytorch import pytorch_specific
from tools import save_and_load


#update gui to fit to default settings
def update_user_input_settings_dict(self):
    # ('Framework', < tkinter.StringVar object at 0x0000014D868EE160 >)
    self.user_input_settings_dict.get('Output').set(self.settings.output_folder)
    self.user_input_settings_dict.get('Train').set(self.settings.folder_trainingsData)
    self.user_input_settings_dict.get('Predict').set(self.settings.folder_to_predict_imgs)
    self.user_input_settings_dict.get('Total Labels').set(self.settings.numberlabels)
    self.user_input_settings_dict.get('Model').set(self.settings.model)
    self.user_input_settings_dict.get('Load old Weights').set(self.settings.loadWeigth)
    self.user_input_settings_dict.get('Split (left/right leg)').set(self.settings.split)
    self.user_input_settings_dict.get('Losses').set(self.settings.loss)
    self.user_input_settings_dict.get('Epochs').set(self.settings.epochs)
    self.user_input_settings_dict.get('Note').set(self.settings.note)
    self.user_input_settings_dict.get('Spatial resolution').set(self.settings.inputShape_create)
    self.user_input_settings_dict.get('Batch size').set(self.settings.batch_size)
    self.user_input_settings_dict.get('Learning rate').set(self.settings.learning_rate)
    get_filenames_in_folder_into_treeview(self, 'Predict')
    get_filenames_in_folder_into_treeview(self, 'Output')
    get_filenames_in_folder_into_treeview(self, 'Train')
    self.update()

#update settings to fit to user input
def update_settings(self):
    # ('Framework', < tkinter.StringVar object at 0x0000014D868EE160 >)
    self.settings.change_output_folder(self.user_input_settings_dict.get('Output').get())
    self.settings.folder_trainingsData = self.user_input_settings_dict.get('Train').get()
    self.settings.folder_to_predict_imgs = self.user_input_settings_dict.get('Predict').get()

    self.settings.numberlabels = self.user_input_settings_dict.get('Total Labels').get()
    self.settings.model = self.user_input_settings_dict.get('Model').get()
    self.settings.loss = self.user_input_settings_dict.get('Losses').get()

    self.settings.split = self.user_input_settings_dict.get('Split (left/right leg)').get()
    self.settings.loadWeigth = self.user_input_settings_dict.get('Load old Weights').get()

    self.settings.epochs = int(self.user_input_settings_dict.get('Epochs').get())
    self.settings.note = self.user_input_settings_dict.get('Note').get()

    if type( self.user_input_settings_dict.get('Spatial resolution').get()) is tuple:
        self.settings.inputShape_create = self.user_input_settings_dict.get('Spatial resolution').get()
    else:
        self.settings.inputShape_create = tuple(map(int, self.user_input_settings_dict.get('Spatial resolution').get().split(' ')))

    self.settings.batch_size = int(self.user_input_settings_dict.get('Batch size').get())
    self.settings.learning_rate = float(self.user_input_settings_dict.get('Learning rate').get())


#get directory
def get_directory(self, category):
    directory = filedialog.askdirectory(title=category)
    self.user_input_settings_dict[category].set(directory)
    get_filenames_in_folder_into_treeview(self, category)

    if category == 'Output':
        json_file = [(root, file)
                     for root, file in self.file_names['Output'].get()
                     if 'settings' in file and file.endswith('.json')
                     ]
        if not json_file:
            self.lbl_text_predict_settings.set('There is a file missing. Need .pt and .json file')
        else:
            self.lbl_text_predict_settings.set('Settings will be taken from \n' + json_file[0][1])
            import_settings_from_json(self, os.path.join(json_file[0][0], json_file[0][1]))
            self.update()

    if self.user_input_settings_dict['Output'].get() != '':
        if self.user_input_settings_dict['Train'].get() != '':
            self.btn_train.configure(state="enable")
        else:
            self.btn_train.configure(state="disable")

        if self.user_input_settings_dict['Predict'].get() != '':
            self.btn_predict.configure(state="enable")
        else:
            self.btn_predict.configure(state="disable")
    else:
        self.btn_train.configure(state="disable")
        self.btn_predict.configure(state="disable")

#left panel filling
def get_filenames_in_folder_into_treeview(self, category):
    directory = self.user_input_settings_dict[category].get()
    result = [(root, name)  # os.path.join(root, name)
              for root, dirs, files in os.walk(directory)
              for name in files
              if name.endswith((".nii", ".gz", ".pt", ".json"))
              ]
    self.file_names[category] = tk.Variable()
    self.file_names[category].set(result)

    index = 1
    self.widgets_dict[category].delete(*self.widgets_dict[category].get_children())

    for root, name in result:
        suffix = name.split('.')[-1]
        if 'label' in name:
            tag = 'label'
        elif suffix == 'pt':
            tag = 'model'
        elif suffix == 'json':
            tag = 'json'
        else:
            tag = 'other'
        self.widgets_dict[category].insert('', END, index,
                                           values=(name, root, suffix), tags=(tag))
        index += 1
    #
    self.widgets_dict[category].tag_configure("label", background='lightgrey', foreground='black')
    self.widgets_dict[category].tag_configure("model", background='lightyellow', foreground='black')
    self.widgets_dict[category].tag_configure("json", background='lightyellow', foreground='black')
    self.widgets_dict[category].tag_configure("other", background='white', foreground='black')

    update_settings(self)
    self.update_idletasks()

#predict
def predict_selected(self):
    self.progressbar_predict['value'] = 0
    self.lbl_predict.grid()
    self.update()
    update_settings(self)
    # self.progressbar.start()
    predictfolder = self.user_input_settings_dict.get('Predict').get()
    outputfolder = self.user_input_settings_dict.get('Output').get()

    if not does_folder_contain_model(outputfolder):
        showerror(title='ERROR', message='There is no model inside the outputfolder')
        return

    if predictfolder != '':
        update_settings(self)
        self.settings.folder_to_predict_imgs = predictfolder
        getFramework().predict(self.settings)
        # self.update_idletasks()
        get_filenames_in_folder_into_treeview(self, 'Predict')
        showinfo("Done", "I predicted!")
        self.lbl_predict.grid_remove()
        get_filenames_in_folder_into_treeview(self, 'Output')
        self.update()

    else:
        showinfo("Error", "Choose a folder from where you want to predict images")

#train
def preprocess_and_train(self):
    self.progressbar_train['value'] = 0
    self.lbl_train.grid()
    self.update()
    print('I would like to train')
    # self.progressbar_train.start()

    trainfolder = self.user_input_settings_dict.get('Train').get()

    if not does_folder_contain_label(trainfolder):
        showerror(title='ERROR', message='There is no label inside the trainfolder')
        return
    update_settings(self)
    ####getdata
    getFramework().save_train_images(self.settings)
    update_settings(self)
    save_and_load.save_settings_as_json(self.settings, GUI_hardcoded.name_json)
    ######train
    try:
        self.lbl_train_text.set("training...")
        self.update()
        getFramework().train(self.settings)
        showinfo("Done", "I trained!")
        # self.progressbar_train.stop()
        self.lbl_train.grid_remove()
        get_filenames_in_folder_into_treeview(self, 'Output')
        self.update_idletasks()

    except Exception:
        showerror(title='ERROR', message=traceback.print_exc())


#get settings from json
def import_settings_from_json(self, json_path):
    with open(json_path, 'r') as j:
        contents = json.loads(j.read())
    for key, value in contents.items():
        setattr(self.settings, key, value)
    self.settings.inputShape_create = tuple(self.settings.inputShape_create)
    update_user_input_settings_dict(self)


def getFramework():
    return pytorch_specific.Pytorch()


def does_folder_contain_label(folder):
    for file in os.listdir(folder):
        if 'label' in file:
            return True
    return False

def does_folder_contain_model(folder):
    for file in os.listdir(folder):
        if str(file).endswith('.pt'):
            return True
    return False

def initial_model_fill(list_of_filenames_in_output,parent):
    for file in list_of_filenames_in_output:
        if file.endswith('.pt'):
            parent.settings.weighted_model_name = file
            parent.settings.model = file
            model_name = file.split('.')[0]
            model_name = model_name.split('_')[0]
            model = load(parent.settings.output_folder + '/' + file, map_location='cpu')
            layer_list = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
            parent.user_input_settings_dict['Total Labels'].set(layer_list[-1].out_channels)
            if model_name in GUI_hardcoded.dropdown_stuff['Model']:
                break
            GUI_hardcoded.dropdown_stuff['Model'].append(model_name)
            menu = parent.widgets_dict['Model']
            menu['menu'].add_command(label=model_name,
                                     command=tk._setit(parent.user_input_settings_dict['Model'], model_name))
            parent.user_input_settings_dict['Model'].set(model_name)
