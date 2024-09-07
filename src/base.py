import numpy as np
from numpy.typing import NDArray
from typing import Callable
import matplotlib.pyplot as plt
from phase_field_2d_ternary.matrix_plot_tools import Ternary


def np_map(func: Callable, arr: NDArray):
    return np.array(list(map(func, arr)))


# %%
def print_shape(*args):
    for elem in args:
        print(f"{elem.shape}")


def plot_as_tile(N, img_list, cmap="gray"):
    fig, axes = plt.subplots(N, N, figsize=(8, 8))
    # 各サブプロットに画像を表示
    for i, ax in enumerate(axes.flat):
        print(i)
        ax.imshow(img_list[i], cmap=cmap)
        ax.axis("off")  # 軸を非表示にする場合


def plot_as_ternary_tile(N, img_list):
    fig, axes = plt.subplots(N, N, figsize=(8, 8))
    # 各サブプロットに画像を表示
    for i, ax in enumerate(axes.flat):
        Ternary.imshow3(img_list[i][0], img_list[i][1])
        ax.axis("off")  # 軸を非表示にする場合
