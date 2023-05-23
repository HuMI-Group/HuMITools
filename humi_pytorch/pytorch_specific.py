import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from framework_abstract_class import Frameworks
from humi_pytorch.Dataset_Mod import ImageDataset
from humi_pytorch.Losses_Mod import fit
from tools.getData import emptyfolder
from tools.save_and_load import save_as_numpy


class Pytorch(Frameworks):

    def convert_result_in_usable_label(self, leg, masks_result):
        return masks_result[leg].argmax(0)

    def get_model_and_weights(self, settings, model=None, weights_path=None):

        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        model = torch.load(os.path.join(settings.folder_model_weights, settings.model + '_model.pt'),
                           map_location=map_location)

        model.eval()

        return model

    def postprocessing_after_predict(self, result, temp_left_all, temp_right_all, scaling_factor_all,
                                     after_split_shape_all, settings, temp_list):
        return super(Pytorch, self).postprocessing_after_predict(result, temp_left_all, temp_right_all,
                                                                 scaling_factor_all,
                                                                 after_split_shape_all, settings, temp_list)

    def save_train_images(self, settings):
        list_processed_img, list_processed_labels = super(Pytorch, self).save_train_images(settings)
        # for humi_pytorch save without one hot encoding and channel only for img, not labels

        aaaaall_images_pre = list_processed_img[..., np.newaxis]

        img_numpy_folder_path = settings.output_folder + '/temp/image_numpy/'
        multiclass_numpy_folder_path = settings.output_folder + '/temp/label_numpy/'

        print('Delete old numpy files...')
        emptyfolder(img_numpy_folder_path)
        emptyfolder(multiclass_numpy_folder_path)

        print('Saving npy files...')
        index_new = 1
        for label, image in zip(list_processed_labels, aaaaall_images_pre):
            # falls labels doch one hot...
            anzahl = int(settings.numberlabels) + 1
            label = np.eye(anzahl)[label]
            save_as_numpy(np.array(image), img_numpy_folder_path, str(index_new))
            save_as_numpy(np.array(label), multiclass_numpy_folder_path, str(index_new))
            index_new += 1

        print('Done with creating npy files for training in humi_pytorch.')

    # overriding abstract method
    def predict(self, settings, model=None, specific_files=None):
        with torch.no_grad():
            super(Pytorch, self).predict(settings)

    def framework_spec_predict(self, settings, preprocessed_img, model):
        inputs = torch.tensor(preprocessed_img).float()
        inputs = inputs.permute(0, 4, 1, 2, 3)
        try:
            inputs = inputs.cuda()
            output = model(inputs)
        except:
            inputs = inputs.cpu()
            output = model(inputs)

        numpy_outp = output.cpu().detach().numpy()
        return numpy_outp

    def train(self, settings):
        if not os.path.exists(settings.output_folder):
            os.makedirs(settings.output_folder)

        # if torch.backends.mps.is_built():
        #    device = torch.device("mps")
        # elif torch.cuda.is_available():
        #    device = torch.device("cuda")
        # else:
        #    device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df = self.create_df(settings)
        print('Total Images: ', len(df))

        testsize = 1 - settings.rel_train_size
        X_train, X_val = train_test_split(df['id'].values, test_size=testsize, random_state=random.randint(1, 20))

        print('Train Size   : ', len(X_train))
        print('Val Size     : ', len(X_val))

        img_numpy_folder_path = settings.output_folder + '/temp/image_numpy/'
        multiclass_numpy_folder_path = settings.output_folder + '/temp/label_numpy/'

        # datasets
        train_set = ImageDataset(img_numpy_folder_path, multiclass_numpy_folder_path, X_train)
        val_set = ImageDataset(img_numpy_folder_path, multiclass_numpy_folder_path, X_val)

        # dataloader
        train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=True)

        model_selected = settings.model.lower()
        from humi_pytorch import model
        model_of_choice = getattr(model, model_selected)
        model = model_of_choice(int(settings.numberlabels))

        weight_decay = 0.00099  ## vlt auch in settings rein?

        criterion = settings.loss
        criterion = criterion.replace(" ", "_").lower()

        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate, weight_decay=weight_decay,
                                     betas=(0.9, 0.9999999), eps=1e-08)  # adamax

        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, settings.learning_rate, epochs=settings.epochs,
                                                    steps_per_epoch=len(train_loader))

        fit(settings, device, settings.epochs, model, train_loader, val_loader, criterion, optimizer, sched)

    def create_df(self, settings):
        name = []
        img_numpy_folder_path = settings.output_folder + '/temp/image_numpy'

        for dirname, _, filenames in os.walk(img_numpy_folder_path):
            for filename in filenames:
                name.append(filename.split('.')[0])

        return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))

    def get_model_forTraining(self, settings):
        pass
