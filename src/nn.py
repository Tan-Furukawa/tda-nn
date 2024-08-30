# %%
from file_operator.yaml_operation import read_yaml
import matplotlib.pyplot as plt
import tensorflow as tf
import img_operation as imop
from tensorflow.keras.datasets import mnist
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

datas = read_yaml("summary/used_in_NN.yaml")
# %%

x0 = np.array(
    list(map(lambda x: [x["information"]["c10"], x["information"]["c20"]], datas))
)


def get_img(img):
    img1 = np.load(img["file_list"][-1][0])
    img2 = np.load(img["file_list"][-1][1])
    n = 64
    # print(img1.shape)
    # print(imop.resize_image(img1, n))
    img = np.stack([imop.resize_image(img1, n), imop.resize_image(img2, n)], axis=2)

    return img


# x0 = list(map(lambda x: np.load(x["persistent_img_path_hom0"]), datas))
x1 = list(map(lambda x: np.load(x["persistent_img_path_hom0"]), datas))

x2 = np.array(list(map(lambda x: get_img(x), datas)))

# x0[0]

# data = datas[0]


d2 = map(lambda x: np.load(x["persistent_img_path_hom1"]), datas)
x3 = np.array(list(d2))
# d2 = map(lambda x: np.load(x["persistent_img_path_hom1"]), datas)
# d2 = map(lambda x: imop.resize_image(imop.trim_image(x, 0.01), 32), d2)

zipped_x = list(zip(x0, x1, x2, x3))
# x1 = np.array(x1)
# x2 = np.array(x2)
# k = np.array()
# k.shape

# x = list(map(lambda x: np.array(
#     np.transpose(
#         [
#             np.load(x["file_list"][-1][0]),
#             np.load(x["file_list"][-1][1])
#         ], (1,2,0))), datas))


# y = list(map(lambda x: [x["information"]["k11"], x["information"]["k12"], x["information"]["k22"]], datas))
y = list(
    map(
        lambda x: [
            x["information"]["w12"],
            x["information"]["w23"],
            x["information"]["w13"],
        ],
        datas,
    )
)
# y = list(
#     map(
#         lambda x: [
#             x["information"]["L12"],
#             x["information"]["L13"],
#             x["information"]["L23"],
#         ],
#         datas,
#     )
# )
# y = list(map(lambda x: [x["information"]["c10"], x["information"]["c20"]], datas))
y = np.array(y)

# トレーニングデータとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(
    zipped_x, y, test_size=0.2, random_state=42
)

x0_train = np.array(list(map(lambda x: x[0], x_train)))
x0_test = np.array(list(map(lambda x: x[0], x_test)))
print(x0_test.shape)
x1_train = np.array(list(map(lambda x: x[1], x_train)))
x1_test = np.array(list(map(lambda x: x[1], x_test)))
print(x1_test.shape)
x2_train = np.array(list(map(lambda x: x[2], x_train)))
x2_test = np.array(list(map(lambda x: x[2], x_test)))
print(x2_test.shape)
x3_train = np.array(list(map(lambda x: x[3], x_train)))
x3_test = np.array(list(map(lambda x: x[3], x_test)))
print(x3_test.shape)
#%%

input0 = Input(shape=x0_train[0].shape)
a0 = layers.Dense(8, activation="relu")(input0)

# # トレーニングデータから検証データをさらに分割
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
input1 = Input(shape=x1_train[0].shape)
# a1 = layers.Flatten()(input1)
a1 = layers.Dense(32, activation="relu")(input1)
a1 = layers.Dense(64, activation="relu")(a1)
a1 = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.1))(a1)
# a1 = layers.Dense(10, activation='relu')(a1)
a1 = layers.Flatten()(a1)

input2 = Input(shape=x2_train[0].shape)
a2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input2)
a2 = layers.MaxPooling2D((2, 2))(a2)
a2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(a2)
a2 = layers.MaxPooling2D((2, 2))(a2)
a2 = layers.Conv2D(
    128,
    (3, 3),
    activation="relu",
    padding="same",
    kernel_regularizer=regularizers.l2(0.1),
)(a2)
# a2 = layers.Conv2D(128, (3, 3), activation='relu')(a2)
a2 = layers.MaxPooling2D((2, 2))(a2)
# a2 = layers.Conv2D(64, (3, 3), activation='relu')(a2)
# a2 = layers.MaxPooling2D((2, 2))(a2)
# a2 = layers.Conv2D(10, (3, 3), activation='relu')(a2)
a2 = layers.Flatten()(a2)


input3 = Input(shape=x3_train[0].shape)
a3 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input3)
a3 = layers.MaxPooling2D((2, 2))(a3)
a3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(a3)
a3 = layers.MaxPooling2D((2, 2))(a3)
a3 = layers.Conv2D(
    128,
    (3, 3),
    activation="relu",
    padding="same",
    kernel_regularizer=regularizers.l2(0.1),
)(a3)
a3 = layers.MaxPooling2D((2, 2))(a3)
a3 = layers.Flatten()(a3)


# 特徴ベクトルを結合
combined = layers.concatenate([a0, a1, a2, a3])

# 全結合層を通じて最終的な出力を生成
z = layers.Dense(64, activation="relu")(combined)
z = layers.Dense(128, activation="relu")(combined)
z = layers.Dense(
    256,
    activation="relu",
    #   kernel_regularizer=regularizers.l2(0.01)
)(combined)
z = layers.Dense(len(y[0]))(z)  # ここではバイナリ分類を例に

# モデルの定義
model = models.Model(inputs=[input0, input1, input2, input3], outputs=z)

# モデルのコンパイル
model.compile(
    optimizer="adam",
    loss="mean_squared_error",  # 回帰問題のため、平均二乗誤差（MSE）を使用
    metrics=["mean_squared_error"],
)  # メトリクスとしてMAEを使用

# モデルのトレーニング
history = model.fit(
    [x0_train, x1_train, x2_train, x3_train],
    y_train,
    epochs=100,
    batch_size=50,
    validation_data=([x0_test, x1_test, x2_test, x3_test], y_test),
)


model.summary()


# モデルの評価
# test_loss, test_acc = model.evaluate([x1_test, x2_test], y_test, verbose=2)
# print(f'\nTest accuracy: {test_acc:.4f}')


predicted = model.predict([x0_test, x1_test, x2_test, x3_test])
expected = y_test
p1 = list(map(lambda x: x[0], predicted))
p2 = list(map(lambda x: x[1], predicted))
# p3 = list(map(lambda x: x[2], predicted))
e1 = list(map(lambda x: x[0], expected))
e2 = list(map(lambda x: x[1], expected))
# e3 = list(map(lambda x: x[2], expected))
#%%
plt.figure(figsize=(4, 4))
# sns.regplot(x=p2, y=e2, s=5)
plt.scatter(p1, e1, s=5)
plt.scatter(p2, e2, s=5)
# plt.scatter(p3, e3, s=5)
r = (np.min([p1]), np.max([p1]))
padding = 0.10
dr = r[1] - r[0]
rr = (r[0] - dr * padding, r[1] + dr * padding)
x = np.linspace(rr[0], rr[1])
plt.xlim(rr[0], rr[1])
plt.ylim(rr[0], rr[1])
plt.xlabel("predicted")
plt.ylabel("expected")
plt.plot(x, x)
plt.show()
# plt.savefig("../gallery/expectation_vs_prediction.pdf")
#%%
# print(predicted)
# print(expected)

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
plt.savefig("../gallery/nn/loss_history.pdf")
# %%

plt.hist(y, bins=10)
# # 訓練と検証のMSEをプロット
# plt.plot(np.array(history.history['mean_squared_error']), label='Training MAE')
# plt.plot(history.history['val_mean_squared_error'], label='Validation MAE')
# plt.title('Model MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
# plt.show()

# %%
