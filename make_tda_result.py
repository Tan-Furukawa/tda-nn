#%%
import numpy as np
import copy
from tda_for_phase_field.random_sampling import (
    random_sampling_from_matrices,
    npMap,
    select_specific_phase,
)

import tda_for_phase_field.tda as tda
import matplotlib.pyplot as plt
from tda_for_phase_field.tda import make_tda_diagram, get_persistent_image_info, plot_persistence_image_from_tda_diagram
from phase_field_2d_ternary.matrix_plot_tools import Ternary
from file_operatoration import extract_directory_information
from file_operator.base import create_directory, save_str
from file_operator.yaml_operation import read_yaml, yaml_dump

import hashlib
import time
import uuid


def generate_unique_filename(head: str="file", extension: str=".npy") -> str:
    timestamp = str(int(time.time()))
    unique_id = str(uuid.uuid4())
    file_hash = hashlib.md5((head + timestamp + unique_id).encode()).hexdigest()

    filename = f"{head}_{file_hash}{extension}"
    return filename

# res1 = extract_directory_information("result")
# res2 = save.dump(res1)
# save.save_str("result_summary.yaml", res2)
#%%
# datas = read_yaml("summary/result_summary_mini.yaml")
datas = read_yaml("summary/result_summary.yaml")
# d_mini = datas[:100]
# d_yaml = yaml_dump(d_mini)
# save_str("summary/result_summary_mini.yaml", d_yaml)
datas_update = copy.deepcopy(datas)

output_file_name = "result_tda"

create_directory(f"summary/{output_file_name}")

#%%

def determine_phase(con1, con2):
    con1_new = np.zeros_like(con1)
    con2_new = np.zeros_like(con2)
    con3 = -con1 - con2 + 1
    con1_new[np.logical_and(con1 >= con3, con1 >= con2)] = 1
    con2_new[np.logical_and(con2 >= con3, con2 > con1)] = 1
    return con1_new, con2_new

total_len = len(datas)

dim0_hole = True

for i, data in enumerate(datas):
    if i % 1000 == 0:
        print(f"progress: {int(i*100 / total_len)}%")
    con1 = np.load(data["file_list"][-1][0])
    con2 = np.load(data["file_list"][-1][1])
    con1, con2 = determine_phase(con1, con2)

    res = random_sampling_from_matrices([con1, con2], 1000)
    phase0 = select_specific_phase(res, 0)
    phase1 = select_specific_phase(res, 1)
    phase2 = select_specific_phase(res, 2)
    threshold = 200

    if (len(phase0) > threshold) and (len(phase1) > threshold) and (len(phase2) > threshold):
        # if j > 10:
        #     break


        print("----------------------------------")
        # Ternary.imshow3(con1, con2)
        # plt.show()
        file_name_hom0 = f"summary/{output_file_name}/" + generate_unique_filename()
        file_name_hom1 = f"summary/{output_file_name}/" + generate_unique_filename()

        datas_update[i]["persistent_img_path_hom0"] = file_name_hom0
        datas_update[i]["persistent_img_path_hom1"] = file_name_hom1

        w = 30

        # save hom 1
        #===================================================
        r1 = make_tda_diagram([phase0, phase1, phase2], dim0_hole=False)
        rr1 = get_persistent_image_info(r1, birth_range=(0,w), pers_range=(0,w), pixel_size=0.5)
        np.save(file_name_hom1, np.array(rr1))


        # save hom 0
        #===================================================
        r0 = make_tda_diagram([phase0, phase1, phase2], dim0_hole=True)

        rr0 = get_persistent_image_info(r0, birth_range=(0,w), pers_range=(0,w), pixel_size=0.5)

        rrr0 = list(map(lambda r: r[0], rr0))
        np.save(file_name_hom0, np.array(rrr0))


        # plt.plot(rr[0][0])
        # plt.show()
        # plt.show()
        # plot_persistence_image_from_tda_diagram(rr[1])
        # plt.show()
        # plot_persistence_image_from_tda_diagram(rr[2])
        # plt.show()

        # plot_persistence_image(rr[0])
        # plt.show()
        # plot_persistence_image(rr[1])
        # plt.show()
        # plot_persistence_image(rr[2])
        # plt.show()
        # tda_res = tda.make_tda_diagram(coordinate_of_points, dim0_hole=False)
    else:
        datas_update[i] = None
        continue
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