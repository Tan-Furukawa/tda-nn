# %%
# ===============================================
# Step 1: ３成分系の計算
# ===============================================
from phase_field_2d_ternary.phase_field import PhaseField2d3c

phase_field = PhaseField2d3c(4, 3, 4, record=True, save_dir_name="../data/ternary/")
phase_field.dtime = 0.005
phase_field.nsave = 300
phase_field.start()

# %%
# ===============================================
# Step 2: Gif画像の描画
# ===============================================
import re
from itertools import groupby
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from phase_field_2d_ternary.matrix_plot_tools import Ternary


def sort_by_number(file_list):
    sorted_file_list = sorted(
        file_list, key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    return sorted_file_list


def pair_files(file_list):
    # 数字部分を取得するための関数
    def get_number(file_name):
        match = re.search(r'con\d+_(\d+)\.npy', file_name)
        return int(match.group(1)) if match else None

    # 数字部分でファイルをソート
    sorted_files = sorted(file_list, key=lambda x: (get_number(x), x))

    # ペアを作成
    paired_files = []
    for _, group in groupby(sorted_files, key=get_number):
        pair = list(group)
        if len(pair) == 2:
            paired_files.append(pair)
    return paired_files

# ディレクトリの中を走査
dir_path = "../data/ternary/output_2024-09-05-14-59-01"
dir_list = os.listdir(dir_path)
dir_list = [f for f in dir_list if re.search(r"\d+", f)]
dir_list = sort_by_number(dir_list)
dir_pair_list = pair_files(dir_list)


images = []
for i, d in enumerate(dir_pair_list):
    if i % 2 == 0:
        continue

    d1 = d[0]
    d2 = d[1]

    con1 = np.load(f"{dir_path}/{d1}")
    con2 = np.load(f"{dir_path}/{d2}")

    fig, ax = plt.subplots()

    #---------------------------------
    # 三角ダイアグラムをプロットするとき(うまくうごかない)
    #---------------------------------
    # phase_field.plot_ternary_contour_and_composition(con1, con2)
    #---------------------------------
    # 組織の変化をプロットするとき
    #---------------------------------
    Ternary.imshow3(con1, con2)

    # 一時的なファイルに保存 (in-memory で保持する)
    fig.canvas.draw()  # Figure を描画
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # PillowのImageオブジェクトに変換してリストに追加
    images.append(Image.fromarray(image))

    plt.close(fig)  # 毎回閉じる

#---------------------------------
# 三角ダイアグラムをプロットするとき(うまくうごかない)
#---------------------------------
images[0].save(
    "animated_image_ternary_diagram_w_4_3_4.gif", save_all=True, append_images=images[1:], duration=100, loop=0
)

#---------------------------------
# 組織の変化をプロットするとき
#---------------------------------
# images[0].save(
#     "animated_image_ternary.gif", save_all=True, append_images=images[1:], duration=100, loop=0
# )
