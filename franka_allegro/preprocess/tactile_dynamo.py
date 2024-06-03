import os 
import cv2
import numpy as np
import pickle 
import h5py
import shutil

from tqdm import tqdm

from .prep_module import PreprocessorModule
from franka_allegro.tactile_data import TactileImage, TactileImageCurved

def dump_tactile_info_to_images(root: str, dump_all=True) -> None:
    # Convert the tactile data to image sequences
    tactile_path = os.path.join(root, 'touch_sensor_values.h5')
    if os.path.exists(os.path.join(root, 'tactile_indices.pkl')):
        print(f'{root} tactile images exist')
        return
    
    os.makedirs(root, exist_ok=True)
    with h5py.File(tactile_path, 'r') as f:
        tactile_timestamps = f['timestamps'][()]
        tactile_fingertip_values= f['fingertip_sensor_values'][()]
        tactile_palm_values = f['palm_sensor_values'][()]
        tactile_finger_values = f['finger_sensor_values'][()]

    with open(os.path.join(root, 'tactile_indices.pkl'), 'wb') as f:
        pickle.dump(
            dict(
                timestamps = tactile_timestamps  
            )
        )
    tactile = dict(
        timestamps = tactile_timestamps,
        fingertip_values = tactile_fingertip_values,
        palm_values = tactile_palm_values,
        finger_values = tactile_finger_values
    )

  
        



class TouchPreprocessor(PreprocessorModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.load_file_name = 'touch_sensor_values.h5'
        self.dump_file_name = 'tactile_indices.pkl'
        self.indices = []

    def __repr__(self):
        return 'touch_reprocessor'

    def load_data(self):
        file_path = os.path.join(self.root, self.load_file_name)
        with h5py.File(file_path, 'r') as f:
            print("Keys: {}".format(list(f.keys())))
            tactile_timestamps = f['timestamps'][()]

        self.data = dict(
            timestamps = tactile_timestamps
        )
        print("Loaded data from {}".format(file_path))


    def get_next_timestamp(self):
        return -1 # Tactile is not considered a 'selective' module at all

    