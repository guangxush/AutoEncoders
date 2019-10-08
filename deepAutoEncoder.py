from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pandas as pd


def deepAutoEncoder():
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input = Input(shape=(4,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(4, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input, decoded)

    encoder = Model(input, encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

    train_dataframe = pd.read_csv('./data/iris.data', header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values
    total_count = 30
    train_level = int(total_count*0.7)

    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    x_test = train_dataset[train_level:, 0:-1].astype('float')
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_output = encoder.predict(x_test)
    encoded_output = decoder.predict(encoded_output)
    print(encoded_output)


if __name__ == '__main__':
    deepAutoEncoder()