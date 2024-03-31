import h5py
import numpy as np

def read_img(file_path):
    """ HDF5ファイルから10チャンネル画像を読み込む """
    with h5py.File(file_path, 'r') as file:
        image = np.array(file['image'])
    return image

def write_img(image, file_path):
    """ 10チャンネル画像をHDF5ファイルとして保存する """
    with h5py.File(file_path, 'w') as file:
        file.create_dataset('image', data=image)

# 使用例
# image = read_multichannel_image('path_to_your_hdf5_file.h5')
# write_multichannel_image(image, 'path_to_save_image.h5')
