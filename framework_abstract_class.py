import os
from abc import ABC, abstractmethod

import nibabel
import numpy as np

from tools import predict
from tools.getData import preprocess_niftis
from tools.predict import undo_reshaping_stuff, merge_overlappings
from tools.save_and_load import save_as_nii_for_control


class Frameworks(ABC):

    @abstractmethod
    def save_train_images(self, settings):
        processed_images, processed_labels = preprocess_niftis(settings)
        return processed_images, processed_labels

    @abstractmethod
    def framework_spec_predict(self, settings, preprocessed_img, model):
        pass

    @abstractmethod
    def get_model_and_weights(self, settings, model=None, weights_path=None):
        pass

    @abstractmethod
    def postprocessing_after_predict(self, result, temp_left_all, temp_right_all, scaling_factor_all,
                                     after_split_shape_all, settings, temp_list):
        print('Merging labels...')
        side = 0
        # label_list = []
        label_list = []
        if settings.split:
            amount_of_images = len(result) // 2
        else:
            amount_of_images = len(result)

        for leg in range(amount_of_images):
            list_to_revert = []

            if settings.split:
                leg_left = side
                label_left_prob = self.convert_result_in_usable_label(leg_left, result)

                leg_rigth = side + 1
                label_right_prob = self.convert_result_in_usable_label(leg_rigth, result)
                side += 2

                list_to_revert.append([leg_left, label_left_prob, temp_left_all[leg]])
                list_to_revert.append([leg_rigth, label_right_prob, temp_right_all[leg]])

            else:
                leg_prob = self.convert_result_in_usable_label(leg, result)
                list_to_revert.append([leg, leg_prob, temp_list[leg]])
            label = undo_reshaping_stuff(after_split_shape_all[leg], list_to_revert, scaling_factor_all,
                                         settings)
            label_list.append(label)
        # overlap
        if label_list.__len__() > 1:
            new_label_list = []
            new_label_pred_list = []
            merge_overlappings(new_label_list, new_label_pred_list, label_list)
            full_label = np.concatenate(new_label_list, axis=2)
        else:
            full_label = np.floor(label_list[0])
        return full_label

    @abstractmethod
    def predict(self, settings, model=None, specific_files=None):
        if model is None:
            model = self.get_model_and_weights(settings)
            model.eval()

        if specific_files is not None:
            all_filesnames = specific_files
        else:
            all_filesnames = [os.path.join(settings.folder_to_predict_imgs, name)
                              for name in os.listdir(settings.folder_to_predict_imgs)
                              if name.endswith((".nii", ".gz")) and 'label' not in name]

        all_predicted_labels = []
        for img_name in all_filesnames:

            plotsavename = img_name.split('\\')[1]
            plotsavename = plotsavename.split('.nii')[0]

            print('Starting prediction for ', plotsavename)

            training_image = nibabel.load(img_name, mmap=False)
            affine = training_image.affine
            q_form = training_image.get_qform()
            image_array_raw = training_image.get_fdata()

            after_split_shape_all, resized_images_list_pre, scaling_factor_all, temp_left_all, temp_list, temp_right_all = predict.preprocess_before_predict(
                image_array_raw, settings)

            result = self.framework_spec_predict(settings, resized_images_list_pre, model)
            label = self.postprocessing_after_predict(result, temp_left_all, temp_right_all, scaling_factor_all,
                                                      after_split_shape_all, settings, temp_list)
            name, nix = img_name.split('\\')[-1].split('.nii')

            modelname = settings.model
            modelname = modelname.split('.pt')[0]
            name = name + '_' + modelname

            savename = name + '-label.nii.gz'
            print('###### Saving merged label as', savename)

            save_as_nii_for_control(label, settings.output_folder, savename, affine, q_form)

            all_predicted_labels.append(label)
            print('-' * 30)
        return all_predicted_labels, all_filesnames

    @abstractmethod
    def convert_result_in_usable_label(self, leg, masks_result):
        pass

    @abstractmethod
    def train(self, settings):
        pass

    @abstractmethod
    def get_model_forTraining(self, settings):
        pass
