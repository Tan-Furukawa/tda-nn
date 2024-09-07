#%%
import numpy as np
from classify_phase import determine_three_phase
import copy
from tda_for_phase_field import SelectPhaseFromSamplingMatrix, PersistentDiagram
import re

from tda_for_phase_field.random_sampling import (
    npMap,
    npFilter
)

import tda_for_phase_field.tda as tda
import matplotlib.pyplot as plt
# from tda_for_phase_field.tda import make_tda_diagram, get_persistent_image_info, plot_persistence_image_from_tda_diagram
from phase_field_2d_ternary.matrix_plot_tools import Ternary
from file_operatoration import extract_directory_information
from file_operator.base import create_directory, save_str
from file_operator.yaml_operation import read_yaml, yaml_dump
from img_operation import trim_image, resize_image


import hashlib
import time
import uuid
#%%


def generate_unique_filename(head: str="file", extension: str=".npy") -> str:
    timestamp = str(int(time.time()))
    unique_id = str(uuid.uuid4())
    file_hash = hashlib.md5((head + timestamp + unique_id).encode()).hexdigest()

    filename = f"{head}_{file_hash}{extension}"
    return filename

def make_minidata(ph_result_path:str = "result", save_as: str="summary/result_summary_mini.yaml")->None:
    # make result_summary_mini.yaml
    #-------------------------------------------
    res1 = extract_directory_information(ph_result_path)
    res2 = yaml_dump(res1[:100])
    save_str(save_as, res2)
    #-------------------------------------------

def extract_number_from_filename(filename):
    # 正規表現で 'con1_' と '.npy' の間の数字を抽出
    match = re.search(r'con1_(\d+)\.npy', filename)
    if match:
        return int(match.group(1))
    else:
        return None


def filter_data(data):
    d = data["file_list"]

    if len(d) == 1:
        return False
    else:
        k = extract_number_from_filename(d[-1][0])
        if k == 120000 or k == 60000:
            return False
        else:
            return True

datas = read_yaml("summary/result_summary.yaml")
datas = npFilter(filter_data, datas)


# datas = read_yaml("summary/result_summary.yaml")
datas_update = copy.deepcopy(datas)
output_file_name = "result_tda2"
#%%
create_directory(f"summary/{output_file_name}")

total_len = len(datas)

dim0_hole = True

reshape_size_hom0 = 30
reshape_size_hom1 = 30


for i, data in enumerate(datas):
    if i % 100 == 0:
        print(f"progress: {int(i*100 / total_len)}%")
    print(data["file_list"][-1][0])
    con1 = np.load(data["file_list"][-1][0])
    con2 = np.load(data["file_list"][-1][1])
    con1, con2 = determine_three_phase(con1, con2)

    phase = SelectPhaseFromSamplingMatrix([con1, con2], 1000)
    x0, y0 = phase.select_specific_phase_as_xy(0)
    p0 = PersistentDiagram(x0, y0)
    hom00, hom10 = p0.get_persistent_image_info(plot=False)

    x1, y1 = phase.select_specific_phase_as_xy(1)
    p1 = PersistentDiagram(x1, y1)
    hom01, hom11 = p1.get_persistent_image_info(plot=False)

    x2, y2 = phase.select_specific_phase_as_xy(2)
    p2 = PersistentDiagram(x2, y2)
    hom02, hom12 = p2.get_persistent_image_info(plot=False)


    _hom0 = np.array([resize_image(hom, reshape_size_hom0) for hom in [hom00, hom01, hom02]])
    hom0 = np.transpose(_hom0, (1,0))

    _hom1 = np.array([resize_image(hom, reshape_size_hom1) for hom in [hom10, hom11, hom12]])
    hom1 = np.transpose(_hom1, (1,2,0))

    # print("----------------------------------")
    # # Ternary.imshow3(con1, con2)
    # # plt.show()
    file_name_hom0 = f"summary/{output_file_name}/" + generate_unique_filename()
    np.save(file_name_hom0, hom0)

    file_name_hom1 = f"summary/{output_file_name}/" + generate_unique_filename()
    np.save(file_name_hom1, hom1)

    datas_update[i]["persistent_img_path_hom0"] = file_name_hom0
    datas_update[i]["persistent_img_path_hom1"] = file_name_hom1

    # w = 30


    # print("-------------------------------")
    # print(i)
    # if i == 36 or i == 43 or i == 44:
    # Ternary.imshow3(con1, con2)
    # plt.show()
    #     plt.savefig(f"../gallery/persistent_img/ternary_exsolution_{i}.pdf")
    #     # plt.show()
    #     plt.imshow(hom0.reshape(30, 1, 3))
    #     plt.savefig(f"../gallery/persistent_img/hom0_diagram_{i}.pdf")
    #     # plt.savefig("gallery/persistent_img/ternary_exsolution.pdf")
    #     # plt.show()
    #     plt.imshow(np.transpose(hom1, (1, 0, 2)), origin="lower")
    #     plt.savefig(f"../gallery/persistent_img/hom1_diagram_{i}.pdf")
    #     # plt.show()

    # print(len(x1))
    # plt.imshow(hom11)
    # plt.colorbar()
    # plt.show()

    # print(len(x2))
    # plt.imshow(hom12)
    # plt.colorbar()
    # plt.show()
    print("-------------------------------")
    # tda_res = tda.make_tda_diagram(coordinate_of_points, dim0_hole=False)
    # else:
    #     datas_update[i] = None
    #     continue
#%%
save_str("summary/used_in_NN.yaml",yaml_dump(list(filter(lambda x: x != None, datas_update))))
#%%

d = read_yaml("summary/used_in_NN.yaml")[0]
dd = np.load(d["persistent_img_path_hom0"])
plt.plot(dd[0])
plt.show()

dd = np.load(d["persistent_img_path_hom1"])[0]
tda.plot_persistence_image_from_tda_diagram(dd)

# coordinate_of_points = npMap(lambda x: [x[0], x[1]], res)
# x = npMap(lambda x: x[0], res)
# y = npMap(lambda x: x[1], res)

# # # tda_res = npFilter(lambda x: x[0] > 5, tda_res)
# # img = tda.get_persistent_image_info(tda_res)
# # tda.plot_persistence_image(img)
# # plt.show()
# # tda.plot_persistent_diagram(coordinate_of_points)
# # plt.show()

# Ternary.imshow3(con1, con2)
# plt.show()
# plt.scatter(y, x, s=2)
# plt.show()

# %%
# con1 = np.load("result/output_2024-08-08-19-32-22/con1_15000.npy")
# con2 = np.load("result/output_2024-08-08-19-32-22/con2_15000.npy")
# #%%
# r = make_tda_diagram([con1, con2, -con1-con2+1], dim0_hole=False)
# rr = get_persistent_image_info(r)
# print(rr)
# plot_persistence_image(rr)