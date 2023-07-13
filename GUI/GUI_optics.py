import os
import sys
import tkinter as tk
from functools import partial
from pathlib import Path
from threading import Thread
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import tools.save_and_load
from GUI import GUI_hardcoded as guistr
from GUI import GUI_functions
from settings import Settings

# path to images/icons
assets_path = Path(__file__).parents[1] / '././assets'

def start_GUI():
    root = tk.Tk()
    root.title('HuMI Tools')
    root.iconbitmap(get_assets_path("icon.ico"))

    win = GUI(master=root)
    window = first_window_outputfolder(win)
    window.grab_set()

    win.pack(fill=BOTH, expand=YES)
    root.protocol("WM_DELETE_WINDOW", lambda: win.on_closing(root))

    root.mainloop()

# set outputfolder initially in an initial popup
class first_window_outputfolder(ttk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.geometry('450x100')
        self.title('Define Outputfolder')
        label = ttk.Label(self,
                          text='Please define your outputfolder. \nModel and .json will be saved there. \nIf model (.py) and .json already exist they will be used for settings')
        label.pack(fill='x', padx=50, pady=5)

        button = ttk.Button(self, text='choose outpufolder', bootstyle="dark-outline",
                            command=lambda category='Output': self.button_set_outputfolder(parent, category))
        button.pack(side=BOTTOM, padx=2, pady=2)

    def button_set_outputfolder(self, parent, category):
        #get directory and set json if available
        GUI_functions.get_directory(parent, category)
        # look if there is already a model.pt
        if GUI_functions.does_folder_contain_model(parent.settings.output_folder):
            list_of_filenames_in_output = os.listdir(parent.settings.output_folder)
            GUI_functions.initial_model_fill(list_of_filenames_in_output, parent)
        else:
            parent.lbl_text_predict_settings.set('You need a model (.pt) to predict')
            parent.update()
        # close popup
        self.destroy()


class GUI(tk.Frame):
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)
        self.logo = tk.PhotoImage(name='logo', file=get_assets_path("logo.png"))

        # fill in dropdown, checkboxes etc...
        self.dropdown_stuff = guistr.dropdown_stuff
        self.path_stuff = guistr.path_stuff
        self.checkbox = guistr.checkbox
        self.text_options = guistr.text_options
        self.advances_options = guistr.advances_options

        self.user_input_settings_dict = {}
        self.widgets_dict = {}
        self.file_names = {}

        self.settings = Settings()
        self.progressbar_predict = ttk.Progressbar()
        self.progressbar_train = ttk.Progressbar()

        self.createGUI()
        self.update_idletasks()
        self.Status = ttk.Label()

    #on closing save settings in .json
    def on_closing(self, root):
        if tk.messagebox.askokcancel("Quit",
                                     "Do you want to quit? \n The .json will be saved or overwritten with the current settings."):
            GUI_functions.update_settings(self)
            tools.save_and_load.save_settings_as_json(self.settings, guistr.name_json)
            root.destroy()

    def createGUI(self):
        # left panel
        left_panel = ttk.Frame(self)
        left_panel.pack(side=LEFT, fill=Y)
        self.create_left_side_preprocessing(left_panel)
        self.create_left_side_train(left_panel)
        self.create_left_side_predict(left_panel)

        # logo
        lbl = ttk.Label(left_panel, image='logo')  # , style='bg.TLabel')
        lbl.pack(side='bottom')

        # right panel
        right_panel = ttk.Frame(self, padding=(2, 1))
        right_panel.pack(side=RIGHT, fill=BOTH, expand=YES)

        ## file input
        for category in self.path_stuff:
            # sep = ttk.Separator(right_panel, bootstyle=SECONDARY)
            # sep.pack(fill=X, padx=2, pady=2)

            self.create_path_input(right_panel, category)

    #right side panel
    def create_path_input(self, right_panel, category):
        browse_frm = ttk.Frame(right_panel)
        browse_frm.pack(side=TOP, fill=X, padx=2, pady=1)

        if category not in self.user_input_settings_dict:
            self.user_input_settings_dict[category] = tk.StringVar(self)

        button = ttk.Button(browse_frm, text=self.path_stuff[category], bootstyle="dark-outline",
                            command=lambda category=category: GUI_functions.get_directory(self, category))
        button.pack(side=LEFT, padx=2, pady=2)

        ######## path label
        label = tk.Label(browse_frm, textvariable=self.user_input_settings_dict[category])
        label.pack(side=LEFT, fill=X, expand=YES)

        browse_frm_tree = ttk.Frame(right_panel)
        browse_frm_tree.pack(side=TOP, fill=X, padx=2, pady=1)
        # ## Treeview
        self.widgets_dict[category] = ttk.Treeview(browse_frm_tree, show='headings', height=10, bootstyle='light',selectmode="none")
        self.widgets_dict[category].pack(side='left', fill=X)  # , pady=1)

        self.widgets_dict[category].configure(columns=('name', 'path', 'type'))
        for col in self.widgets_dict[category]['columns']:
            self.widgets_dict[category].heading(col, text=col.title(), anchor=W)

        self.yscrollbar = ttk.Scrollbar(browse_frm_tree, orient='vertical', bootstyle='secondary')
        self.yscrollbar.pack(side='right', fill=Y)

        self.widgets_dict[category].configure(yscrollcommand=self.yscrollbar.set)
        self.widgets_dict[category].config(yscrollcommand=self.yscrollbar.set)
        self.yscrollbar.config(command=self.widgets_dict[category].yview)

    def create_left_side_preprocessing(self, left_panel):
        train_frame = LeftFrame(left_panel)
        train_frame.pack(fill=BOTH, pady=1)
        ## container
        status_frm = ttk.Frame(train_frame, padding=10)
        status_frm.columnconfigure(1, weight=1)
        train_frame.add(
            child=status_frm,
            title='General Settings',
            bootstyle="light"
        )

        ## settings stuff
        index_row = 0
        paddings = {'padx': 5, 'pady': 5}
        category = guistr.str_split
        self.create_checkbox(status_frm, category, index_row, paddings)
        index_row += 1

        category = guistr.str_totallabels
        input = self.text_options[category]
        self.create_entrybox(status_frm, category, index_row, paddings, input)
        index_row += 1

        # advances settings
        cf = CollapsingFrame(status_frm)
        cf.grid(row=index_row, column=0, padx=1, pady=1, rowspan=2, columnspan=2, sticky=tk.EW)
        group1 = ttk.Frame(cf, padding=10)

        index_row += 1
        category = guistr.str_spatialres
        input = self.advances_options[category]

        self.create_entrybox(group1, category, index_row, paddings, input)
        index_row += 1
        cf.add(group1, title='Advanced settings', style='light')

        index_row += 1

    def create_left_side_train(self, left_panel):
        train_frame = LeftFrame(left_panel)
        train_frame.pack(fill=BOTH, pady=1)
        ## container
        status_frm = ttk.Frame(train_frame, padding=10)
        status_frm.columnconfigure(1, weight=1)
        train_frame.add(
            child=status_frm,
            title='Training',
            bootstyle='info'
        )
        ## settings stuff
        index_row = 0
        paddings = {'padx': 5, 'pady': 5}
        for category in self.dropdown_stuff:
            self.create_dropdown(status_frm, category, index_row, paddings)
            index_row += 1


        for category in self.text_options:
            if category == guistr.str_totallabels:
                continue
            input = self.text_options[category]
            self.create_entrybox(status_frm, category, index_row, paddings, input)
            index_row += 1

        category = guistr.str_loadweights
        self.create_checkbox(status_frm, category, index_row, paddings)
        index_row += 1

        # advances settings
        cf = CollapsingFrame(status_frm)
        cf.grid(row=index_row, column=0, padx=1, pady=1, rowspan=2, columnspan=2, sticky=tk.EW)
        group1 = ttk.Frame(cf, padding=10)

        for category in self.advances_options:
            if category == guistr.str_spatialres:
                continue
            index_row += 1
            input = self.advances_options[category]

            self.create_entrybox(group1, category, index_row, paddings, input)
            index_row += 1
        cf.add(group1, title='Advanced settings', style='light')

        index_row += 1
        self.btn_train = ttk.Button(
            master=status_frm,
            text='Train',
            compound=LEFT,
            command=partial(GUI_functions.preprocess_and_train, self),
            bootstyle='info'
        )

        self.btn_train.configure(state="disabled")
        self.btn_train.grid(row=index_row, column=0, columnspan=2, sticky=E, **paddings)
        train_frame.pack(fill=BOTH, pady=1)

        index_row += 1

        ## progress message
        self.lbl_train_text = tk.StringVar()
        self.lbl_train_text.set("preprocessing...")
        self.lbl_train = ttk.Label(
            master=status_frm,
            textvariable=self.lbl_train_text,
            font='Helvetica 10 bold'
        )
        self.lbl_train.grid(row=index_row, column=0, columnspan=2, sticky=W)
        self.lbl_train.grid_remove()
        index_row += 1

        ## progress bar
        self.progressbar_train = ttk.Progressbar(
            master=status_frm,
            mode='determinate',
            # variable='prog-value',
            bootstyle='info'
        )
        self.progressbar_train.grid(row=index_row, column=0, columnspan=2, sticky=EW, pady=(10, 5))

    #create predict
    def create_left_side_predict(self, left_panel):
        paddings = {'padx': 5, 'pady': 5}

        predict_frame = LeftFrame(left_panel)
        predict_frame.pack(fill=BOTH, pady=1)

        status_frm = ttk.Frame(predict_frame, padding=10)
        status_frm.columnconfigure(1, weight=1)
        predict_frame.add(
            child=status_frm,
            title='Prediction',
            bootstyle='info'
        )
        self.lbl_text_predict_settings = tk.StringVar()
        self.lbl_text_predict_settings.set('Select an outputfolder with a .json')

        # predict button
        self.btn_predict = ttk.Button(
            master=status_frm,
            text='Predict',
            compound=LEFT,
            command=partial(GUI_functions.predict_selected, self),
            bootstyle='info'
        )
        self.btn_predict.configure(state="disabled")
        self.btn_predict.grid(row=7, column=0, columnspan=2, sticky=E, **paddings)

        ## progress message
        self.lbl_predict = ttk.Label(
            master=status_frm,
            text='predicting...',
            font='Helvetica 10 bold'
        )
        self.lbl_predict.grid(row=8, column=0, columnspan=2, sticky=W)
        self.lbl_predict.grid_remove()
        ## progress bar
        self.progressbar_predict = ttk.Progressbar(
            master=status_frm,
            mode='determinate',
            # variable='prog-value',
            bootstyle='info'
        )
        self.progressbar_predict.grid(row=9, column=0, columnspan=2, sticky=EW, pady=(10, 5))

    def create_dropdown(self, master, category, index_row, paddings):
        # Creating a unique variable / name for later reference
        self.user_input_settings_dict[category] = tk.StringVar()
        self.user_input_settings_dict[category].set(self.dropdown_stuff[category][0])
        label = tk.Label(master, text=category)
        label.grid(column=0, row=index_row, sticky=tk.W, **paddings)
        # Creating OptionMenu with unique variable
        self.widgets_dict[category] = ttk.OptionMenu(master, self.user_input_settings_dict[category],
                                                     self.dropdown_stuff[category][0],
                                                     *self.dropdown_stuff[category], bootstyle='dark-outline', )
        self.widgets_dict[category].grid(row=index_row, column=1, padx=1, pady=1, sticky=tk.EW)
        index_row += 1
        return index_row

    def create_entrybox(self, master, category, index_row, paddings, input):
        self.user_input_settings_dict[category] = tk.Variable()
        self.user_input_settings_dict[category].set(input)
        label = tk.Label(master, text=category)
        label.grid(column=0, row=index_row, sticky=tk.W, **paddings)
        # Creating OptionMenu with unique variable
        self.widgets_dict[category] = ttk.Entry(master, textvariable=self.user_input_settings_dict[category],
                                                bootstyle='dark')
        self.widgets_dict[category].grid(row=index_row, column=1, padx=1, pady=1, sticky=tk.EW)
        # self.create_tooltip(self.widgets_dict[category], self.tooltiptext[category])
        index_row += 1
        return index_row

    def create_checkbox(self, master, category, index_row, paddings):
        self.user_input_settings_dict[category] = tk.BooleanVar()
        self.user_input_settings_dict[category].set(self.checkbox[category])
        self.widgets_dict[category] = ttk.Checkbutton(master, text=category,
                                                      variable=self.user_input_settings_dict[category],
                                                      bootstyle='dark')
        self.widgets_dict[category].grid(column=0, row=index_row, sticky=tk.EW, **paddings)
        index_row += 1
        return index_row


class LeftFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)

    def add(self, child, title="", bootstyle=INFO, **kwargs):
        style_color = bootstyle
        frm = ttk.Frame(self, bootstyle=style_color)
        frm.grid(sticky=EW)  # row=self.cumulative_rows, column=0, )

        # header title
        header = ttk.Label(
            master=frm,
            text=title,
            bootstyle=(style_color, INVERSE)
        )
        if kwargs.get('textvariable'):
            header.configure(textvariable=kwargs.get('textvariable'))
        header.pack(side=LEFT, fill=BOTH, padx=10)

        # header toggle button
        btn = ttk.Button(
            master=frm,
            # image=self.images[0],
            bootstyle=style_color,
            # command=_func
        )
        btn.pack(side=RIGHT)

        # assign toggle button to child so that it can be toggled
        child.btn = btn
        child.grid(sticky=NSEW)


#get path for assets
def get_assets_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    else:
        return assets_path / filename


#for advanced settings
class CollapsingFrame(ttk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.columnconfigure(0, weight=1)
        self.cumulative_rows = 0
        self.images = [ttk.PhotoImage(name='open', file=get_assets_path("arrow_up.png")),
                       ttk.PhotoImage(name='closed', file=get_assets_path("arrow_down.png"))]


    def add(self, child, title="", style='light', **kwargs):
        if child.winfo_class() != 'TFrame':
            return
        style_color = style.split('.')[0]
        frm = ttk.Frame(self, style=f'{style_color}.TFrame')
        frm.grid(row=self.cumulative_rows, column=0, sticky='ew')
        lbl = ttk.Label(frm, text=title, style='light.inverse.TLabel')
        if kwargs.get('textvariable'):
            lbl.configure(textvariable=kwargs.get('textvariable'))
        lbl.pack(side='left', fill='both', padx=10)
        btn = ttk.Button(frm, image='open', style='light', command=lambda c=child: self._toggle_open_close(child))

        btn.pack(side='right')
        child.btn = btn
        child.grid(row=self.cumulative_rows + 1, column=0, sticky='news')
        self.cumulative_rows += 2
        #initially closed
        child.grid_remove()
        child.btn.configure(image='closed')

    def _toggle_open_close(self, child):
        if child.winfo_viewable():
            child.grid_remove()
            child.btn.configure(image='closed')
        else:
            child.grid()
            child.btn.configure(image='open')

