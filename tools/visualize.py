import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from tools import save_and_load


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[2] // 2
    ax.imshow(volume[:, :, ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[:, :, ax.index])
    if len(ax.images) > 1:
        ax.images[1].set_array(ax.label[:, :, ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[:, :, ax.index])
    if len(ax.images) > 1:
        ax.images[1].set_array(ax.label[:, :, ax.index])


def show_center_overlay(image, label):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax1 = plt.subplots()
    ax1.volume = image
    ax1.label = label
    ax1.index = image.shape[2] // 2
    ax1.imshow(image[:, :, ax1.index], cmap='gray')
    cmap = colors.ListedColormap(
        ['#000000', '#80ae80', '#b17a65', '#d8654f', '#90ee90', '#dcf514', '#fffadc', '#c8c8eb'])
    ax1.imshow(label[:, :, ax1.index], cmap=cmap, alpha=0.5)
    fig.canvas.mpl_connect('key_press_event', process_key)
    return fig


def plotLogger(input):
    for model in input.models:
        folder = input.folder_model_weights + model + '_Results'
        input.folder = folder
        for file in os.listdir(folder):
            if file.endswith("_logger.csv"):
                model = file.split('_logger')[0]
                path_file = os.path.join(folder, file)
                csv_file = pd.read_csv(path_file)
                plt.plot(csv_file['loss'], label=model + '_training')
                plt.plot(csv_file['val_loss'], label=model + 'validation')

    plt.title('Model performance')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper right")
    plt.savefig(input.folder_model_weights + 'Losses.png')


def overlay_in_2d(image, label, name, input):
    plt.ioff()
    plt.figure()
    middle = int(image.shape[2] / 2)
    plt.imshow(image[:, :, middle], 'gray', interpolation='none')
    cmap = colors.ListedColormap(
        ['#000000', '#80ae80', '#b17a65', '#d8654f', '#90ee90', '#dcf514', '#fffadc', '#c8c8eb'])
    plt.imshow(label[:, :, middle], cmap=cmap, interpolation='none', alpha=0.7)
    if hasattr(input, 'epoch'):
        name = name + '#' + input.epoch
    plt.savefig(input.folder + '/' + input.model + '/img/' + name + '.png')


def intersect_label(all_label_raw, all_label_pred, all_name, input, affine_list):
    print('-' * 30)
    print('Creating intersetions...')
    print('-' * 30)
    for label_raw, label_pred, name, affine in zip(all_label_raw, all_label_pred, all_name, affine_list):
        intersect_label = np.zeros(label_raw.shape)
        for index, value in np.ndenumerate(label_raw):
            value_raw = label_raw[index]
            value_pred = label_pred[index]
            if value_raw == value_pred:
                intersect_label[index] = value_raw
            elif value_pred == 0:
                # missing
                intersect_label[index] = 100
            elif value_raw == 0:
                # dazu
                intersect_label[index] = 68
            else:
                # mix
                intersect_label[index] = 300 + value_pred

        name = name.split('-label')[0]
        save_name = input.folder + '/' + input.model + '/intersect/' + name + '_intersect-label.nii'
        save_and_load.save_as_nii_for_control(intersect_label, save_name, affine=affine)
    print('-' * 30)
    print('Saved all intersetions...')
    print('-' * 30)
