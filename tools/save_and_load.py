import datetime
import json
import os
from datetime import datetime

import nibabel
import numpy as np

import settings


def save_as_nii_for_control(array, path, filename, affine=None, q_form=None):
    if path == '':
        path_filename = filename
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        path_filename = os.path.join(path, filename)
    if affine is None:
        if not os.path.isfile(settings.affine_example_filename):
            settings.affine_example_filename = '.' + settings.affine_example_filename
        example_img_for_affine = nibabel.load(settings.affine_example_filename, mmap=False)
        affine = example_img_for_affine.affine
    nii = nibabel.Nifti1Image(np.array(array), affine=affine)
    if q_form is not None:
        nii.set_qform(q_form)
    nibabel.save(nii, path_filename)


def save_as_numpy(array, path, filename):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path + filename), array)
    except:
        print('failed')


def save_settings_as_json(settings, file_name):
    import tkinter

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            # print(obj)
            if isinstance(obj, datetime):
                return str(obj)
            if isinstance(obj, tkinter.ttk.Progressbar):
                return str(obj)
            if obj.__str__().__contains__('gui'):  #
                return str(obj)
            return super(NpEncoder, self).default(obj)

    dict_settings = settings.__dict__
    with open(os.path.join(settings.output_folder, file_name), 'w') as f:
        json.dump(dict_settings, f, cls=NpEncoder)
