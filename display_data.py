#%%
from file_operator.yaml_operation import read_yaml
import numpy as np
from phase_field_2d_ternary.matrix_plot_tools import Ternary
import matplotlib.pyplot as plt
from tda_for_phase_field.tda import make_tda_diagram, get_persistent_image_info, plot_persistence_image_from_tda_diagram

datas = read_yaml("summary/used_in_NN.yaml")
# x = list(map(lambda x: np.transpose(np.load(x["persistent_img_path"]), (1,2,0)), datas))


for i, data in enumerate(datas):
    if i > 100:
        break
    con1 = np.load(data["file_list"][-1][0])
    con2 = np.load(data["file_list"][-1][1])
    Ternary.imshow3(con1, con2)

    d = np.load(data["persistent_img_path"])
    plot_persistence_image_from_tda_diagram(d[0])
    plt.show()
#%%