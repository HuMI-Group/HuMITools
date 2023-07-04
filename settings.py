class Settings:
    def __init__(self):
        self.rel_train_size = 0.80
        self.save_predict_next_to_img = True

        # for train
        self.loadWeigth = False
        self.epochs = 300
        self.split_legs = False
        self.number_of_labels = 1

        # manuel
        self.inputShape_create = (144, 144, 22)  # (144, 144, 22)#(104,104,22) ###spatial resolution
        self.batch_size = 4
        self.learning_rate = 0.00001

        self.labels_left = []
        self.labels_right = []

    def change_output_folder(self, output_folder):
        self.output_folder = output_folder
        self.folder_model_weights = output_folder
        self.path_images_train_single_numpy = output_folder + '/temp/image_numpy/'
        self.path_labels_train_multiclass_numpy = output_folder + '/temp/label_numpy/'

    def error_stop(self, message):
        print('#### ERROR:', message)


affine_example_filename = './assets/08_20170320_GW_UMC_AB_Dixon_01_T1_iDEAL.nii'
