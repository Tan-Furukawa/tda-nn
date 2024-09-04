# %%
from file_operator.yaml_operation import read_yaml
from base import np_map, print_shape
from photo_operation import convert_to_binary_img
import matplotlib.pyplot as plt
import tensorflow as tf
import img_operation as imop
from tensorflow.keras.datasets import mnist
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.datasets import mnist

import nn_operation as nno

datas = read_yaml("summary/used_in_NN_2d_fix.yaml")

def filter_dat(dat):
    img = np.load(dat["resized_img"])
    if np.std(img) < 0.01:
        return False
    else:
        return True
datas = list(filter(filter_dat, datas))

get_c0 = lambda x: [x["information"]["c0"]]
# get_c0 = lambda x: [0]
x0 = nno.pick_up_datas(get_c0, datas)


def get_img(dat):
    img = np.load(dat["resized_img"])
    n = 64
    dat = imop.resize_image(img, n)
    return dat
    # img1 = np.load(img["file_list"][-1][0])
    # img2 = np.load(img["file_list"][-1][1])
    # n = 64
    # img = np.stack([imop.resize_image(img1, n), imop.resize_image(img2, n)], axis=2)

    # return img


get_hom0_img = lambda x: imop.resize_image(np.load(x["persistent_img_path_hom0"]), 38)

x1 = nno.pick_up_datas(get_hom0_img, datas)


def get_img_for_binary(dat):
    img = np.load(dat["resized_img"])
    img = convert_to_binary_img(img)
    n = 64
    dat = imop.resize_image(img, n)
    #  変換後にconvertする場合
    dat = convert_to_binary_img(dat)
    return dat


x2 = nno.pick_up_datas(get_img_for_binary, datas)
x2 = np.expand_dims(x2, axis=-1)


#%%

#%%
# plt.imshow(x2[2], cmap="gray")
# plt.axis("off")
# plt.savefig("../gallery/teacher_data/1_1_binary-3.pdf")
# # プロット全体を調整して表示
# plt.tight_layout()
# plt.show()
# for i, x in enumerate(x2):
#     if i > 100:
#         break
#     plt.imshow(x)
#     plt.show()

#%%


get_hom1_img = lambda x: imop.resize_image(np.load(x["persistent_img_path_hom1"]), 38)
x3 = nno.pick_up_datas(get_hom1_img, datas)
x3 = np.expand_dims(x3, axis=-1)

zipped_x = list(zip(x0, x1, x2, x3))
# zipped_x = list(zip(x0, x1, x3))
# zipped_x = list(zip(x0, x1, x2))


get_w = lambda x: [x["information"]["w"]]
y = nno.pick_up_datas(get_w, datas)


(x_train, y_train), (x_validation, y_validation), (x_test, y_test) = (
    nno.divide_data_as_test_valid_train(zipped_x, y)
)

x_train_unzip = nno.unzip(x_train)
x_validation_unzip = nno.unzip(x_validation)
x_test = nno.unzip(x_test)

print_shape(*x_train_unzip)


nn_block_fn_list = [
    nno.make_mini_lnn_block,
    nno.make_lnn_block,
    nno.make_cnn_block,
    nno.make_cnn_block,
]

input_list, block_list = nno.apply_nn_to_x(x_train_unzip, nn_block_fn_list)

z = nno.make_lnn_combined_block(block_list, y)
# z = nno.make_lnn_combined_block([a0, a1, a3], y)

model = models.Model(inputs=input_list, outputs=z)

from tensorflow.keras.utils import plot_model
plot_model(model)
#%%

# model = models.Model(inputs=[input0, input1, input3], outputs=z)

# モデルのコンパイル
model.compile(
    optimizer="adam",
    loss="mean_squared_error",  # 回帰問題のため、平均二乗誤差（MSE）を使用
    metrics=["mean_squared_error"],
)  # メトリクスとしてMAEを使用

# モデルのトレーニング
history = model.fit(
    x_train_unzip,
    y_train,
    epochs=200,
    batch_size=50,
    validation_data=(x_validation_unzip, y_validation),
)

model.summary()


# モデルの評価
# test_loss, test_acc = model.evaluate([x1_test, x2_test], y_test, verbose=2)
# print(f'\nTest accuracy: {test_acc:.4f}')


predicted = model.predict(x_validation_unzip)
expected = y_validation
p1 = list(map(lambda x: x[0], predicted))
# p2 = list(map(lambda x: x[1], predicted))
# p3 = list(map(lambda x: x[2], predicted))
e1 = list(map(lambda x: x[0], expected))
# e2 = list(map(lambda x: x[1], expected))
# e3 = list(map(lambda x: x[2], expected))
# %%
plt.figure(figsize=(4, 4))
# sns.regplot(x=p2, y=e2, s=5)
plt.scatter(p1, e1, s=5)
# plt.scatter(p2, e2, s=5)
# plt.scatter(p3, e3, s=5)
r = (np.min([p1]), np.max([p1]))
padding = 0.10
dr = r[1] - r[0]
rr = (r[0] - dr * padding, r[1] + dr * padding)
x = np.linspace(rr[0], rr[1])
plt.xlim(rr[0], rr[1])
plt.ylim(rr[0], rr[1])
plt.xlabel("predicted w")
plt.ylabel("expected w")
plt.plot(x, x)
# plt.savefig("../gallery/nn_binary/w.pdf")
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(4, 4))
plt.plot(np.log(np.array(history.history["loss"])) ** 1 / 2, label="Training Loss")
plt.plot(
    np.log(np.array(history.history["val_loss"])) ** 1 / 2, label="Validation Loss"
)
# plt.ylim((0,np.log(np.max(history.history['val_loss'][5:]))))
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("log(loss)")
plt.legend()
# plt.savefig("../gallery/nn_binary/loss_history_eta22.pdf")
# %%
from photo_operation import image_to_grayscale_array_pil, convert_to_binary_img

image_to_grayscale_array_pil("photo_data/rect11305.png")

# plt.hist(y, bins=10)
# # 訓練と検証のMSEをプロット
# plt.plot(np.array(history.history['mean_squared_error']), label='Training MAE')
# plt.plot(history.history['val_mean_squared_error'], label='Validation MAE')
# plt.title('Model MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.show()

# %%
