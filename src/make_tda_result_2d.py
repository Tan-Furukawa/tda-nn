# %%
import numpy as np
from classify_phase import determine_three_phase
import copy
from tda_for_phase_field import SelectPhaseFromSamplingMatrix, PersistentDiagram
import re

from tda_for_phase_field.random_sampling import npMap, npFilter
import tda_for_phase_field.tda as tda
import matplotlib.pyplot as plt

from phase_field_2d_ternary.matrix_plot_tools import Ternary
from file_operatoration import extract_directory_information
from file_operator.base import create_directory, save_str
from file_operator.yaml_operation import read_yaml, yaml_dump
from img_operation import trim_image, resize_image, expand_image


import hashlib
import time
import uuid

# %%


def generate_unique_filename(head: str = "file", extension: str = ".npy") -> str:
    timestamp = str(int(time.time()))
    unique_id = str(uuid.uuid4())
    file_hash = hashlib.md5((head + timestamp + unique_id).encode()).hexdigest()

    filename = f"{head}_{file_hash}{extension}"
    return filename


def make_minidata(
    ph_result_path: str = "result", save_as: str = "summary/result_2d_summary_mini.yaml"
) -> None:
    # make result_summary_mini.yaml
    # -------------------------------------------
    res1 = extract_directory_information(ph_result_path)
    res2 = yaml_dump(res1[:100])
    save_str(save_as, res2)
    # -------------------------------------------


def extract_number_from_filename(filename):
    # 正規表現で 'con1_' と '.npy' の間の数字を抽出
    match = re.search(r"con1_(\d+)\.npy", filename)
    if match:
        return int(match.group(1))
    else:
        return None


def filter_data(data):
    return True
    # d = data["file_list"]

    # if len(d) == 1:
    #     return False
    # else:
    #     k = extract_number_from_filename(d[-1][0])
    #     if k == 120000 or k == 60000:
    #         return False
    #     else:
    #         return True


datas = read_yaml("summary/result_2d_summary_fix.yaml")
datas = npFilter(filter_data, datas)


# datas = read_yaml("summary/result_summary.yaml")
datas_update = copy.deepcopy(datas)
output_file_name = "result_2d_tda_fix"
# %%
create_directory(f"summary/{output_file_name}")

total_len = len(datas)

dim0_hole = True

reshape_size_hom0 = 30
reshape_size_hom1 = 30

# %%

for i, data in enumerate(datas):
    # if i > 1:
    #     break
    if i % 10 == 0:
        print(f"progress: {int(i*100 / total_len)}%")
    print(data["file_list"][0][-1])
    con = np.load(data["file_list"][-1][0])
    r = np.random.uniform(0.5, 1)
    con = expand_image(con, r)
    # plt.imshow(con)
    # plt.show()
    phase = SelectPhaseFromSamplingMatrix([con], 2000)
    x0, y0 = phase.select_specific_phase_as_xy(0)
    # plt.scatter(x0, y0, s=1)
    # plt.show()
    # con1, con2 = determine_three_phase(con1, con2)
    # x0, y0 = phase.select_specific_phase_as_xy(0)
    p0 = PersistentDiagram(x0, y0)
    hom00, hom10 = p0.get_persistent_image_info(plot=False)
    # plt.imshow(hom10)
    # plt.show()

    file_name_con = f"summary/{output_file_name}/" + generate_unique_filename()
    np.save(file_name_con, con)

    file_name_hom0 = f"summary/{output_file_name}/" + generate_unique_filename()
    np.save(file_name_hom0, hom00)

    file_name_hom1 = f"summary/{output_file_name}/" + generate_unique_filename()
    np.save(file_name_hom1, hom10)

    datas_update[i]["resized_img"] = file_name_con
    datas_update[i]["persistent_img_path_hom0"] = file_name_hom0
    datas_update[i]["persistent_img_path_hom1"] = file_name_hom1

    print("-------------------------------")
# %%
save_str(
    "summary/used_in_NN_2d_fix.yaml",
    yaml_dump(list(filter(lambda x: x != None, datas_update))),
)
# %%

# # d = read_yaml("summary/used_in_NN_2d_fix.yaml")
# from itertools import product

# import time
# import numpy as np
# from sklearn import datasets
# from scipy.stats import multivariate_normal as mvn
# import matplotlib.pyplot as plt

# from ripser import Rips
# from persim import PersistenceImager

# con = np.load(d[111]["resized_img"])
# # con = con[:50, :50]
# phase = SelectPhaseFromSamplingMatrix([con], 3000)
# x0, y0 = phase.select_specific_phase_as_xy(0)

# data = np.stack([y0, x0], axis=1)
# # data = np.flip(data)

# rips = Rips()
# dgms = rips.fit_transform(data)
# H0_dgm = dgms[0]
# H1_dgm = dgms[1]

# plt.imshow(con, )
# plt.figure(figsize=(4,4))
# # plt.scatter(data[:,0], data[:,1], s=4)
# plt.scatter(data[:,0], data[:,1], s=4)
# plt.savefig("scatter_img.pdf")
#%%

plt.figure(figsize=(4,4))
rips.plot(dgms, legend=False, show=False)
# plt.savefig("persistence_diagram_h0_h1.pdf")
# plt.title("Persistence diagram of $H_0$ and $H_1$")
# plt.show()
# p0 = PersistentDiagram(x0, y0)
# hom00, hom10 = p0.get_persistent_image_info(plot=False)

# p0.hom1_diagram

# dd = np.expand_dims(dd, axis=-1)

# plt.imshow(dd)
# plt.savefig("../gallery/hom_img/hom1.pdf")

#%%

# from photo_operation import convert_to_binary_img
# dd = np.load(d[111]["resized_img"])
# dd = convert_to_binary_img(dd)
# plt.imshow(dd, cmap="gray")
# plt.colorbar()
# plt.savefig("../gallery/hom_img/img_binary.pdf")

# plt.show()
# dd = np.load(d["persistent_img_path_hom1"])[0]

# coordinate_of_points = npMap(lambda x: [x[0], x[1]], res)
# x = npMap(lambda x: x[0], res)
# y = npMap(lambda x: x[1], res)
