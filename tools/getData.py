import os

import nibabel
import numpy as np

from tools import cropping_and_resizing, splitting
from tools.save_and_load import save_as_nii_for_control


def preprocess_niftis(settings):
    print('-' * 30)
    print('Creating training data...')

    ############grab the correct folders for creating the training dataset
    path = settings.folder_trainingsData

    all_filesnames = [os.path.join(path, name)
                      for name in os.listdir(path)
                      if name.endswith((".nii", ".gz"))]

    if not all_filesnames:
        settings.error_stop('no files')

    list_image_names = []
    images_train_list = []
    label_train_list = []

    # sort label and image filenames
    filenames_images = []
    filenames_labels = []

    for filename in all_filesnames:
        if 'label.nii' in filename or 'mask.nii' in filename or 'Mask.nii' in filename:
            filenames_labels.append(filename)
        else:
            filenames_images.append(filename)

    filenames_images = sorted(filenames_images)
    filenames_labels = sorted(filenames_labels)
    list_image_names.append(filenames_images)
    # get label and image arrays,
    # split into packages around the size of settings.inputShape_create(2)
    # normalize image from 0-100
    normalized_img_in_packages = []
    normalized_labls_in_packages = []
    for label, image in zip(filenames_labels, filenames_images):
        print('---')
        print('Label: ' + label)
        print('Image: ' + image)
        training_mask = nibabel.load(label, mmap=False)
        training_image = nibabel.load(image, mmap=False)
        label_array = np.array(training_mask.get_fdata(), dtype=np.uint8)
        image_array = training_image.get_fdata()

        if image_array.shape == label_array.shape:
            rounded_times_split = image_array.shape[2] // settings.inputShape_create[2]
            if rounded_times_split != 0:
                list_splittet_image_arrays = np.array_split(image_array, rounded_times_split, axis=2)
                list_splittet_label_arrays = np.array_split(label_array, rounded_times_split, axis=2)
            else:
                list_splittet_image_arrays = [image_array]
                list_splittet_label_arrays = [label_array]

            for image, label in zip(list_splittet_image_arrays, list_splittet_label_arrays):
                # Range normalize
                image *= 100.0 / image.max()

                normalized_img_in_packages.append(image)
                normalized_labls_in_packages.append(label)
        else:
            print('##### Skipped image because of shape issue: ', image)

    # split left and right // or not
    # convert counting to classes
    loop = 0
    for label_array, image_array in zip(normalized_labls_in_packages, normalized_img_in_packages):
        # split left and right leg
        to_process_label_list = []
        to_process_image_list = []
        if settings.split:

            image_array_left, image_array_right, label_array_left, label_array_right = splitting.split_image_and_label(
                image_array, label_array, settings)

            label_array_left, label_array_right = convert_labels_to_start_from_1(label_array_left,
                                                                                 label_array_right, settings)
            to_process_label_list.append(label_array_left)
            to_process_image_list.append(image_array_left)
            # mirror
            to_process_label_list.append(label_array_right[::-1, :, :])
            to_process_image_list.append(image_array_right[::-1, :, :])

        else:
            to_process_label_list.append(label_array)
            to_process_image_list.append(image_array)

        # cut background
        for to_process_image, to_process_label in zip(to_process_image_list, to_process_label_list):
            processed_image_left, processed_label_left = cropping_and_resizing.crop_image_and_label_background(
                to_process_image, to_process_label)

            # append
            label_train_list.append(processed_label_left)
            images_train_list.append(processed_image_left)

        loop = loop + 1

    print('Resize to largest dim...')
    # zoom to settings.inputShape_create
    processed_images, processed_labels = cropping_and_resizing.resize_list_img_and_label_to_given_dim(
        images_train_list, label_train_list, settings)

    print('-' * 30)
    print('Shape of all images: ' + str(processed_images[0].shape))
    save_as_nii_for_control(processed_labels, settings.output_folder + '/temp/', 'list-processed-label')
    save_as_nii_for_control(processed_images, settings.output_folder + '/temp/', 'list-processed-img')
    save_as_nii_for_control(processed_labels[0], settings.output_folder + '/temp/', 'processed-label')
    save_as_nii_for_control(processed_images[0], settings.output_folder + '/temp/', 'processed-img')
    print('Saved some test images --> processed-img and processed-label plus lists')

    return np.array(processed_images), np.array(processed_labels)


def emptyfolder(foldername):
    if os.path.exists(foldername):
        for numpy in os.scandir(foldername):
            try:
                if os.path.isfile(numpy) or os.path.islink(numpy):
                    os.unlink(numpy)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (numpy, e))


def convert_labels_to_start_from_1(label_left, label_right, settings):
    index = 0

    #### if middle-finding also outputs one more label from other side
    if np.unique(label_left).__len__() != np.unique(label_right).__len__():
        print('Maybe split did not work correct, but there is an uneven number of labels on left and right side')

    settings.labels_left = np.unique(label_left).tolist()
    settings.labels_right = np.unique(label_right).tolist()

    label_right_new = np.zeros(label_right.shape, dtype=np.uint8)
    label_left_new = np.zeros(label_left.shape, dtype=np.uint8)

    for old_value_left, old_value_right in zip(np.unique(label_left), np.unique(label_right)):
        label_left_new[label_left == old_value_left] = index
        label_right_new[label_right == old_value_right] = index
        index += 1
        if index > int(settings.numberlabels):
            break

    return label_left_new, label_right_new
