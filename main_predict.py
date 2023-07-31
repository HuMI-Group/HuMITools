import sys
import os

from humi_pytorch import pytorch_specific
from settings import Settings
from tools import save_and_load
json_path = sys.argv[1]

if os.path.isfile(json_path):
    print('Taking the following .json: ', json_path)
    settings = Settings()
    save_and_load.load_from_json_to_settings(json_path, settings)
    print(settings.labels_left)
    pytorch_specific.Pytorch().predict(settings)
else:
    print('The file does not exist')

