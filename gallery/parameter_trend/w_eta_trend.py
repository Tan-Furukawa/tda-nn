# %%

# wとetaのを変化させると組織がどのようになるのかを描画する
#===========================================================
# Step 1: データ作成
#===========================================================

from phase_field_2d import PhaseField
import matplotlib.pyplot as plt
import numpy as np

w = np.linspace(2, 4, num=8)
eta = np.linspace(1, 3, num=8)

w, eta = np.meshgrid(w, eta)
w = w.flatten()
eta = eta.flatten()

parameters = np.stack([w, eta], axis=1)

for p in parameters:
    phase_field = PhaseField(
        p[0],
        0.4,
        method="anisotropic",
        eta=np.array([1.0, p[1], 0.0]),
        save_dir_name="../data/w_eta_param",
        record=True,
    )

    phase_field.nprint = 10000
    phase_field.nsave = 2000
    phase_field.nstep = 10000
    phase_field.dtime = 0.005
    phase_field.seed = 1
    phase_field.start()

# %%
#===========================================================
# Step 2: 描画
#===========================================================

from file_operator.get_files_from_dir import numerical_sort_by_underscore
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import Normalize

from phase_field_2d_ternary.matrix_plot_tools import get_matrix_image


def sort_files_by_date(file_list):
    # 日付を解析してソートする
    sorted_files = sorted(
        file_list, key=lambda x: datetime.strptime(x, "output_%Y-%m-%d-%H-%M-%S")
    )
    return sorted_files


dir_path = "../data/w_eta_param"
dir_list = os.listdir(dir_path)

sorted_files = sort_files_by_date(dir_list)

img_list = list(map(lambda x: np.load(f"{dir_path}/{x}/con_10000.npy"), sorted_files))

N = 8
fig, axes = plt.subplots(N, N, figsize=(8, 8))
# 各サブプロットに画像を表示
for i, ax in enumerate(axes.flat):
    cmap = plt.get_cmap("Grays")
    # カラーマップの正規化
    norm = Normalize(vmin=0, vmax=1)
    ax.imshow(img_list[i], cmap="gray", norm=norm)
    ax.axis("off")  # 軸を非表示にする場合

plt.savefig("img_result.pdf")
# %%
