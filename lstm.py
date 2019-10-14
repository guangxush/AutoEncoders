import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Dense, Input, LSTM, RepeatVector
import pandas as pd


def demo():
    # 数据预处理
    train_dataframe = pd.read_csv('./data/iris.data', header=0)
    train_dataset = train_dataframe.values
    total_count = 150
    train_level = int(total_count*0.7)

    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    x_test = train_dataset[train_level:, 0:-1].astype('float')
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    print(x_train.shape)
    print(x_test.shape)

    # parameters for LSTM
    n_out_seq_length = 4
    input_dim = 1

    # 压缩特征维度至5维
    encoding_dim = 2
    # this is our input placeholder
    data_input = Input(shape=(4, 1, ), name="data_input")
    decoder_input = Input(shape=(encoding_dim, ), name="decoder_input")

    # 编码层
    encoded_1 = LSTM(units=encoding_dim, name='encoded_1')(data_input)

    # 解码层
    decoded_1 = RepeatVector(n_out_seq_length, name='decoded_1')(encoded_1)
    decoded_2 = LSTM(input_dim, return_sequences=True, activation=None, name='decoded_2')(decoded_1)

    # 构建自编码模型
    autoencoder = Model(inputs=data_input, outputs=decoded_2)

    # 构建编码模型
    encoder = Model(inputs=data_input, outputs=encoded_1)

    # 构建解码模型
    _decoded_1 = autoencoder.get_layer(name='decoded_1')(decoder_input)
    _decoded_2 = autoencoder.get_layer(name='decoded_2')(_decoded_1)
    decoder = Model(inputs=decoder_input, outputs=_decoded_2)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, epochs=30, batch_size=10, shuffle=True)

    encoded_iris = encoder.predict(x_test)
    print(encoded_iris)

    decoded_iris = decoder.predict(encoded_iris)
    print(decoded_iris)

    autoencoder = autoencoder.predict(x_test)
    autoencoder = autoencoder.flatten()
    print(autoencoder)


if __name__ == '__main__':
    demo()