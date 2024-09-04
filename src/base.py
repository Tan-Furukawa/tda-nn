import numpy as np
from numpy.typing import NDArray
from typing import Callable

def np_map(func: Callable, arr: NDArray):
    return np.array(list(map(func, arr)))

#%%
def print_shape (*args):
    for elem in args:
        print(f"{elem.shape}")

def plot_as_tile(N, img_list)
    fig, axes = plt.subplots(N, N, figsize=(8, 8))
    # 各サブプロットに画像を表示
    for i, ax in enumerate(axes.flat):
        print(i)
        ax.imshow(img_list[i], cmap='gray')
        ax.axis('off')  # 軸を非表示にする場合