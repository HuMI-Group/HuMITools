import os
import tkinter as tk
import traceback
import json
from tkinter import filedialog
from tkinter.constants import END
from tkinter.messagebox import showinfo, showerror
from torch import load
from torch import nn

from GUI import GUI_hardcoded as guistr
from humi_pytorch import pytorch_specific
from tools import save_and_load


#update gui to fit to default settings
def update_user_input_settings_dict(self):
    self.user_input_settings_dict.get(guistr.str_output).set(self.settings.output_folder)
    self.user_input_settings_dict.get(guistr.str_train).set(self.settings.folder_trainingsData)
    self.user_input_settings_dict.get(guistr.str_predict).set(self.settings.folder_to_predict_imgs)
    self.user_input_settings_dict.get(guistr.str_totallabels).set(self.settings.number_of_labels)
    self.user_input_settings_dict.get(guistr.str_model).set(self.settings.model)
    self.user_input_settings_dict.get(guistr.str_loadweights).set(self.settings.loadWeigth)
    self.user_input_settings_dict.get(guistr.str_split).set(self.settings.split_leftright)
    self.user_input_settings_dict.get(guistr.str_losses).set(self.settings.loss)
    self.user_input_settings_dict.get(guistr.str_epochs).set(self.settings.epochs)
    self.user_input_settings_dict.get(guistr.str_spatialres).set(self.settings.inputShape_create)
    self.user_input_settings_dict.get(guistr.str_batchsize).set(self.settings.batch_size)
    self.user_input_settings_dict.get(guistr.str_learningrate).set(self.settings.learning_rate)
    self.user_input_settings_dict.get(guistr.str_labelsright).set(self.settings.labels_right)
    self.user_input_settings_dict.get(guistr.str_labelsleft).set(self.settings.labels_left)

    get_filenames_in_folder_into_treeview(self, guistr.str_predict)
    get_filenames_in_folder_into_treeview(self, guistr.str_output)
    get_filenames_in_folder_into_treeview(self, guistr.str_train)
    self.update()


#update settings to fit to user input
def update_settings(self):
    self.settings.change_output_folder(self.user_input_settings_dict.get(guistr.str_output).get())
    self.settings.folder_trainingsData = self.user_input_settings_dict.get(guistr.str_train).get()
    self.settings.folder_to_predict_imgs = self.user_input_settings_dict.get(guistr.str_predict).get()

    self.settings.number_of_labels = self.user_input_settings_dict.get(guistr.str_totallabels).get()
    self.settings.model = self.user_input_settings_dict.get(guistr.str_model).get()
    self.settings.loss = self.user_input_settings_dict.get(guistr.str_losses).get()

    self.settings.split_leftright = self.user_input_settings_dict.get(guistr.str_split).get()
    self.settings.loadWeigth = self.user_input_settings_dict.get(guistr.str_loadweights).get()

    self.settings.epochs = int(self.user_input_settings_dict.get(guistr.str_epochs).get())
    self.settings.batch_size = int(self.user_input_settings_dict.get(guistr.str_batchsize).get())
    self.settings.learning_rate = float(self.user_input_settings_dict.get(guistr.str_learningrate).get())

    if type(self.user_input_settings_dict.get(guistr.str_spatialres).get()) is tuple:
        self.settings.inputShape_create = self.user_input_settings_dict.get(guistr.str_spatialres).get()
    else:
        self.settings.inputShape_create = tuple(map(int, self.user_input_settings_dict.get(guistr.str_spatialres).get().split(' ')))

    if self.user_input_settings_dict.get(guistr.str_labelsright).get() != '':
        self.settings.labels_right = list(self.user_input_settings_dict.get(guistr.str_labelsright).get())
        self.settings.labels_left = list(self.user_input_settings_dict.get(guistr.str_labelsleft).get())


#get directory
def get_directory(self, category):
    directory = filedialog.askdirectory(title=category)
    self.user_input_settings_dict[category].set(directory)
    get_filenames_in_folder_into_treeview(self, category)

    if category == guistr.str_output:
        json_file = [(root, file)
                     for root, file in self.file_names[guistr.str_output].get()
                     if 'settings' in file and file.endswith('.json')
                     ]
        if json_file:
            import_settings_from_json(self, os.path.join(json_file[0][0], json_file[0][1]))
            self.update()

    if self.user_input_settings_dict[guistr.str_output].get() != '':
        if self.user_input_settings_dict[guistr.str_train].get() != '':
            self.btn_train.configure(state="enable")
        else:
            self.btn_train.configure(state="disable")

        if self.user_input_settings_dict[guistr.str_predict].get() != '' and self.user_input_settings_dict[guistr.str_output].get():
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
    self.widgets_dict[category].tag_configure("label", background='#f2f2f2', foreground='black')
    self.widgets_dict[category].tag_configure("model", background='white', foreground='#5bc0de')
    self.widgets_dict[category].tag_configure("json", background='white', foreground='#5bc0de')
    self.widgets_dict[category].tag_configure("other", background='white', foreground='black')

    update_settings(self)
    self.update_idletasks()


#predict
def predict_selected(self):
    predictfolder = self.user_input_settings_dict.get(guistr.str_predict).get()
    outputfolder = self.user_input_settings_dict.get(guistr.str_output).get()

    if not does_folder_contain_model(outputfolder):
        showerror(title='ERROR', message='There is no model inside the outputfolder')
        return
    if predictfolder == '':
        showinfo(title="Error", message="Choose a folder from where you want to predict images")
        return

    self.progressbar_predict['value'] = 0
    self.lbl_predict.grid()
    self.update()
    update_settings(self)

    try:
        update_settings(self)
        self.settings.folder_to_predict_imgs = predictfolder
        getFramework().predict(self.settings)
        get_filenames_in_folder_into_treeview(self, guistr.str_predict)
        showinfo("Done", "I predicted!")
        self.lbl_predict.grid_remove()
        get_filenames_in_folder_into_treeview(self, guistr.str_output)
        self.update()
    except (BaseException):
        print(traceback.format_exc())
        showerror(title='ERROR - Check console', message=traceback.format_exc().splitlines()[-1])
        self.lbl_predict.grid_remove()

#train
def preprocess_and_train(self):
    trainfolder = self.user_input_settings_dict.get(guistr.str_train).get()
    if not does_folder_contain_label(trainfolder):
        showerror(title='ERROR', message='There is no label inside the trainfolder')
        return

    self.progressbar_train['value'] = 0
    self.lbl_train.grid()
    self.update()
    # self.progressbar_train.start()
    update_settings(self)
    ####getdata
    try:
        getFramework().save_train_images(self.settings)
        update_settings(self)
        save_and_load.save_settings_as_json(self.settings, guistr.name_json)
    ######train
        self.lbl_train_text.set("training...")
        self.update()
        getFramework().train(self.settings)
        showinfo("Done", "I trained!")
        # self.progressbar_train.stop()
        self.lbl_train.grid_remove()
        get_filenames_in_folder_into_treeview(self, guistr.str_output)
        self.update_idletasks()
    except (BaseException):
        print(traceback.format_exc())
        showerror(title='ERROR - Check console', message=traceback.format_exc().splitlines()[-1])
        self.lbl_train.grid_remove()


#get settings from json
def import_settings_from_json(self, json_path):
    save_and_load.load_from_json_to_settings(json_path,self.settings)
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
            model = load(parent.settings.output_folder + '/' + file, map_location='cpu')
            layer_list = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
            try:
                output_layer = int(layer_list[-1].out_channels)-1
            except:
                output_layer = int(layer_list[-2].out_channels) - 1
            parent.user_input_settings_dict[guistr.str_totallabels].set(output_layer)
            if model_name in guistr.dropdown_stuff[guistr.str_model]:
                parent.user_input_settings_dict[guistr.str_model].set(model_name)
                break
            guistr.dropdown_stuff[guistr.str_model].append(model_name)
            menu = parent.widgets_dict[guistr.str_model]
            menu['menu'].add_command(label=model_name,
                                     command=tk._setit(parent.user_input_settings_dict[guistr.str_model], model_name))
            parent.user_input_settings_dict[guistr.str_model].set(model_name)
            parent.user_input_settings_dict[guistr.str_loadweights].set(True)

