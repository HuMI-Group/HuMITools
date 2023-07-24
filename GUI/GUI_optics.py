import os
import sys
import tkinter as tk
from functools import partial
from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from GUI.GUI_tooltip import ToolTip

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
            # parent.lbl_text_predict_settings.set('You need a model (.pt) to predict')
            parent.update()
        # close popup
        self.destroy()


class GUI(tk.Frame):
    def __init__(self, master=None, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)
        self.logo = tk.PhotoImage(name='logo', file=get_assets_path("logo.png"))
        self.tooltiptext = guistr.tooltiptext

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
        # Initialize style
        labelframe_general_settings = tk.Frame(self, highlightbackground="#E1F8DC", highlightthickness=4)
        labelframe_general_settings.grid(row=0, column=0, columnspan=2, sticky='news', padx=5, pady=5)

        header_frame = tk.Frame(labelframe_general_settings)
        header_frame.grid(column=0, row=0, columnspan=2, sticky=NSEW)
        header_frame.columnconfigure(1, weight=1)
        header = ttk.Label(header_frame, text=u"General Settings", background='#E1F8DC', font=('Tahoma', 20), padding=(5,0))
        header.grid(column=0, row=0, columnspan=3, sticky=NSEW)

        frame_general_settings_left = ttk.Frame(labelframe_general_settings)
        frame_general_settings_left.grid(row=1, column=0, sticky='news', padx=10, pady=5)
        frame_general_settings_left.columnconfigure(1, weight=1)

        frame_general_settings_right = ttk.Frame(labelframe_general_settings)
        frame_general_settings_right.grid(row=1, column=1, sticky='news', padx=5, pady=5)
        frame_general_settings_right.columnconfigure(1, weight=1)

        self.create_left_side_preprocessing(frame_general_settings_left)
        self.create_path_input(frame_general_settings_right, guistr.str_output)

        #train
        labelframe_train = tk.Frame(self, highlightbackground="#CAF1DE", highlightthickness=4)
        labelframe_train.grid(row=1, column=0, columnspan=2, sticky='news', padx=5, pady=5)

        header_frame = tk.Frame(labelframe_train)
        header_frame.grid(column=0, row=0, columnspan=3, sticky=NSEW)
        header_frame.columnconfigure(1, weight=1)

        header = ttk.Label(header_frame, text=u"Training", background='#CAF1DE', font=('Tahoma', 20),padding=(5,0))
        header.grid(column=0, row=0, columnspan=3, sticky=NSEW)

        labelframe_train_left = ttk.Frame(labelframe_train)
        labelframe_train_left.grid(row=1, column=0, sticky='news', padx=10, pady=5)
        labelframe_train_right = ttk.Frame(labelframe_train)
        labelframe_train_right.grid(row=1, column=1, sticky='news', padx=5, pady=5)

        self.create_left_side_train(labelframe_train_left)
        self.create_path_input(labelframe_train_right, guistr.str_train)

        #predict
        labelframe_predict = tk.Frame(self, highlightbackground="#ACDDDE", highlightthickness=4)
        labelframe_predict.grid(row=2, column=0, columnspan=2, sticky='news', padx=5, pady=5)
        labelframe_predict.columnconfigure(0, weight=1)

        header_frame = tk.Frame(labelframe_predict)
        header_frame.grid(column=0, row=0, columnspan=3, sticky=NSEW)
        header_frame.columnconfigure(1, weight=1)

        header = ttk.Label(header_frame, text=u"Prediction", background='#ACDDDE', font=('Tahoma', 20),padding=(5,0))
        header.grid(column=0, row=0, columnspan=3, sticky=NSEW)

        labelframe_predict_left = ttk.Frame(labelframe_predict)
        labelframe_predict_left.grid(row=1, column=0, sticky='news', padx=10, pady=5)
        labelframe_predict_left.columnconfigure(1, weight=1)

        labelframe_predict_right = ttk.Frame(labelframe_predict)
        labelframe_predict_right.grid(row=1, column=1, sticky='news', padx=5, pady=5)
        labelframe_predict_right.columnconfigure(1, weight=1)

        self.create_left_side_predict(labelframe_predict_left)
        self.create_path_input(labelframe_predict_right, guistr.str_predict)

    def create_tooltip(self,widget, text):
        toolTip = ToolTip(widget)

        def enter(event):
            toolTip.showtip(text)

        def leave(event):
            toolTip.hidetip()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    #right side panel
    def create_path_input(self, right_panel, category):
        browse_frm = ttk.Frame(right_panel)
        browse_frm.grid(row=0, column=0, sticky='news', padx=2, pady=2)

        if category not in self.user_input_settings_dict:
            self.user_input_settings_dict[category] = tk.StringVar(self)

        button = ttk.Button(browse_frm, text=self.path_stuff[category], bootstyle="dark-outline",
                            command=lambda category=category: GUI_functions.get_directory(self, category))
        button.grid(row=0, column=0, sticky='w', padx=2, pady=2)

        ######## path label
        label = tk.Label(browse_frm, textvariable=self.user_input_settings_dict[category])
        label.grid(row=1, column=0, sticky='e', padx=2, pady=2)

        # ## Treeview
        self.widgets_dict[category] = ttk.Treeview(right_panel, show='headings', height=10, bootstyle='light',selectmode="none")

        self.widgets_dict[category].grid(row=2, column=0, sticky='news', padx=2, pady=2)

        self.widgets_dict[category].configure(columns=('name'))
        for col in self.widgets_dict[category]['columns']:
            self.widgets_dict[category].heading(col, text=col.title(), anchor=W)
            self.widgets_dict[category].column(col, minwidth=0, width=500, stretch=NO)

        self.yscrollbar = ttk.Scrollbar(right_panel, orient='vertical', bootstyle='secondary')
        self.yscrollbar.grid(row=2 ,column=1, sticky='ns')

        self.widgets_dict[category].configure(yscrollcommand=self.yscrollbar.set)
        self.widgets_dict[category].config(yscrollcommand=self.yscrollbar.set)
        self.yscrollbar.config(command=self.widgets_dict[category].yview)

    def create_left_side_preprocessing(self, left_panel):
        input_frame = ttk.Frame(left_panel)
        input_frame.grid(column=0, row=1, columnspan=3,sticky=EW,pady=5)
        ## settings stuff
        index_row = 1
        paddings = {'padx': 5, 'pady': 5}
        category = guistr.str_split
        self.create_checkbox(input_frame, category, index_row, paddings)
        index_row += 1

        category = guistr.str_totallabels
        input = self.text_options[category]
        self.create_entrybox(input_frame, category, index_row, paddings, input)
        index_row += 1

        # advances settings
        cf = CollapsingFrame(input_frame)
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
        input_frame = ttk.Frame(left_panel)
        input_frame.grid(column=0, row=1, columnspan=3, sticky=EW,pady=5)
        ## settings stuff
        index_row = 0
        paddings = {'padx': 5, 'pady': 5}
        for category in self.dropdown_stuff:
            self.create_dropdown(input_frame, category, index_row, paddings)
            index_row += 1

        for category in self.text_options:
            if category == guistr.str_totallabels:
                continue
            input = self.text_options[category]
            self.create_entrybox(input_frame, category, index_row, paddings, input)
            index_row += 1

        category = guistr.str_loadweights
        self.create_checkbox(input_frame, category, index_row, paddings)
        index_row += 1

        # advances settings
        cf = CollapsingFrame(input_frame)
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
            master=input_frame,
            text='Train',
            compound=LEFT,
            command=partial(GUI_functions.preprocess_and_train, self),
            bootstyle='info'
        )

        self.btn_train.configure(state="disabled")
        self.btn_train.grid(row=index_row, column=0, columnspan=2, sticky=E, **paddings)
        self.create_tooltip(self.btn_train, self.tooltiptext['str_trainbutton'])

        index_row += 1

        ## progress message
        self.lbl_train_text = tk.StringVar()
        self.lbl_train_text.set("preprocessing...")
        self.lbl_train = ttk.Label(
            master=input_frame,
            textvariable=self.lbl_train_text,
            font='Helvetica 14 bold'
        )
        self.lbl_train.grid(row=index_row, column=0, columnspan=2, sticky=W)
        self.lbl_train.grid_remove()
        index_row += 1

        ## progress bar
        self.progressbar_train = ttk.Progressbar(
            master=input_frame,
            mode='determinate',
            # variable='prog-value',
            bootstyle='info'
        )
        self.progressbar_train.grid(row=index_row, column=0, columnspan=2, sticky=EW, pady=(10, 5))

    #create predict
    def create_left_side_predict(self, left_panel):
        # input_frame = ttk.Frame(left_panel)
        # input_frame.grid(column=0, row=1, columnspan=3, sticky=EW, pady=5)
        # input_frame.
        ## settings stuff
        paddings = {'padx': 5, 'pady': 5}

        # predict button
        self.btn_predict = ttk.Button(
            master=left_panel,
            text='Predict',
            compound=LEFT,
            command=partial(GUI_functions.predict_selected, self),
            bootstyle='info'
        )
        self.btn_predict.configure(state="disabled")
        self.btn_predict.grid(row=1, column=3, columnspan=3, sticky=E, **paddings)
        self.create_tooltip(self.btn_predict, self.tooltiptext['str_predictbutton'])

        ## progress message
        self.lbl_predict = ttk.Label(
            master=left_panel,
            text='predicting...',
            font='Helvetica 14 bold'
        )
        self.lbl_predict.grid(row=2, column=0, columnspan=2, sticky=W)
        self.lbl_predict.grid_remove()
        ## progress bar
        self.progressbar_predict = ttk.Progressbar(
            master=left_panel,
            mode='determinate',
            # variable='prog-value',
            bootstyle='info'
        )
        self.progressbar_predict.grid(row=3, column=0, columnspan=6, sticky=EW, pady=(10, 5))

        # logo
        lbl = ttk.Label(left_panel, image='logo')  # , style='bg.TLabel')
        lbl.grid(row=4, column=1)

    def create_dropdown(self, master, category, index_row, paddings):
        # Creating a unique variable / name for later reference
        self.user_input_settings_dict[category] = tk.StringVar()
        self.user_input_settings_dict[category].set(self.dropdown_stuff[category][0])
        label = ttk.Label(master, text=category, font=('Tahoma', 10))
        label.grid(column=0, row=index_row, sticky=tk.W, **paddings)
        # Creating OptionMenu with unique variable
        self.widgets_dict[category] = ttk.OptionMenu(master, self.user_input_settings_dict[category],
                                                     self.dropdown_stuff[category][0],
                                                     *self.dropdown_stuff[category], bootstyle='dark-outline')
        self.widgets_dict[category].grid(row=index_row, column=1, padx=1, pady=1, sticky=tk.EW)
        self.widgets_dict[category].configure(width=20)

          # add this to your code it will change style for dropdownlist

        self.create_tooltip(self.widgets_dict[category], self.tooltiptext[category])

        index_row += 1
        return index_row

    def create_entrybox(self, master, category, index_row, paddings, input):
        self.user_input_settings_dict[category] = tk.Variable()
        self.user_input_settings_dict[category].set(input)
        label = ttk.Label(master, text=category, font=('Tahoma', 10))
        label.grid(column=0, row=index_row, sticky=tk.W, **paddings)
        # Creating OptionMenu with unique variable
        self.widgets_dict[category] = ttk.Entry(master, textvariable=self.user_input_settings_dict[category],
                                                bootstyle='dark')
        self.widgets_dict[category].grid(row=index_row, column=1, padx=1, pady=1, sticky=tk.EW)
        self.create_tooltip(self.widgets_dict[category], self.tooltiptext[category])
        index_row += 1
        return index_row

    def create_checkbox(self, master, category, index_row, paddings):
        style = ttk.Style()
        style.configure("myButton.TCheckbutton", font=("Tahoma", 10))

        self.user_input_settings_dict[category] = tk.BooleanVar()
        self.user_input_settings_dict[category].set(self.checkbox[category])
        self.widgets_dict[category] = ttk.Checkbutton(master, text=category,
                                                      variable=self.user_input_settings_dict[category],
                                                      style='myButton.TCheckbutton')
        self.widgets_dict[category].grid(column=0, row=index_row, sticky=tk.EW, **paddings)
        self.create_tooltip(self.widgets_dict[category], self.tooltiptext[category])

        index_row += 1
        return index_row


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

