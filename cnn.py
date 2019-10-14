import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, Flatten
import pandas as pd


def demo():
    # 数据预处理
    train_dataframe = pd.read_csv('./data/iris.data', header=0)
    train_dataset = train_dataframe.values
    total_count = 150
    train_level = int(total_count * 0.7)

    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    x_test = train_dataset[train_level:, 0:-1].astype('float')
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_train_output = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape)
    print(x_test.shape)

    # 压缩特征维度至2维
    encoding_dim = 2
    # this is our input placeholder
    data_input = Input(shape=(4, 1,), name="data_input")
    decoder_input = Input(shape=(encoding_dim,), name="decoder_input")

    # 编码层
    encoded_1 = Conv1D(filters=16, kernel_size=2, padding='same', activation='relu', strides=1,
                       name='encoder_1')(data_input)
    encoded_2 = MaxPooling1D(pool_size=2, name='encoder_2')(encoded_1)
    encoder_3 = Flatten(name='encoder_3')(encoded_2)
    encoded_4 = Dense(10, activation='relu', name='encoded_4')(encoder_3)
    encoded_5 = Dense(encoding_dim, name='encoded_5')(encoded_4)

    # 解码层
    decoded_1 = Dense(10, activation='relu', name='decode_1')(encoded_5)
    decoded_2 = Dense(64, activation='relu', name='decoded_2')(decoded_1)
    decoded_3 = Dense(128, activation='relu', name='decoded_3')(decoded_2)
    decoded_4 = Dense(4, name='decoded_4')(decoded_3)

    # 构建自编码模型
    autoencoder = Model(inputs=data_input, outputs=decoded_4)
    autoencoder.summary()

    # 构建编码模型
    encoder = Model(inputs=data_input, outputs=encoded_5)

    # 构建解码模型
    _decoded_1 = autoencoder.get_layer(name='decode_1')(decoder_input)
    _decoded_2 = autoencoder.get_layer(name='decoded_2')(_decoded_1)
    _decoded_3 = autoencoder.get_layer(name='decoded_3')(_decoded_2)
    _decoded_4 = autoencoder.get_layer(name='decoded_4')(_decoded_3)
    decoder = Model(inputs=decoder_input, outputs=_decoded_4)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train_output, epochs=100, batch_size=10, shuffle=True)

    encoded_iris = encoder.predict(x_test)
    print(encoded_iris)

    decoded_iris = decoder.predict(encoded_iris)
    print(decoded_iris)

    autoencoder_img = autoencoder.predict(x_test)
    print(autoencoder_img)


if __name__ == '__main__':
    demo()
