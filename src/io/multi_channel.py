import h5py
import numpy as np
from typing import Union
from pathlib import Path
import numpy.typing as npt

def read_img(file_path: Union[str, Path]) -> npt.NDArray[np.float64]:
    """ HDF5ファイルから10チャンネル画像を読み込む """
    with h5py.File(file_path, 'r') as file:
        image = np.array(file['image'])
    return image

def write_img(image: npt.NDArray[np.float64], file_path: Union[str, Path]) -> None:
    """ 10チャンネル画像をHDF5ファイルとして保存する """
    with h5py.File(file_path, 'w') as file:
        file.create_dataset('image', data=image)

# 使用例
# image = read_multichannel_image('path_to_your_hdf5_file.h5')
# write_multichannel_image(image, 'path_to_save_image.h5')
