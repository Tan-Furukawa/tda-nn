# %%
from file_operator.yaml_operation import read_yaml
from base import np_map, print_shape
from photo_operation import convert_to_binary_img
import matplotlib.pyplot as plt
import tensorflow as tf
import img_operation as imop
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.datasets import mnist
import nn_operation as nno
from phase_field_2d_ternary.matrix_plot_tools import Ternary

# datas = read_yaml("summary/used_in_NN_2d_fix.yaml")
datas = read_yaml("summary/used_in_NN.yaml")


def get_ternary_img(img):
    # img = np.load(dat["resized_img"])
    # n = 64
    # dat = imop.resize_image(img, n)
    # return dat
    img1 = np.load(img["file_list"][-1][0])
    img2 = np.load(img["file_list"][-1][1])
    n = 64
    img = np.stack([imop.resize_image(img1, n), imop.resize_image(img2, n)], axis=2)

    return img1, img2


tmp = nno.pick_up_datas(get_ternary_img, datas)

#%%
for i in range(40):
    Ternary.imshow3(tmp[i][0], tmp[i][1])
    plt.axis("off")  # 軸を非表示にする場合
    plt.savefig(f"../gallery/nn/ternary_{i}.pdf")
    # plt.show()

# from base import plot_as_ternary_tile

# plot_as_ternary_tile(5, tmp)

# %%
