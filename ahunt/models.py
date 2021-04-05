from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

INIT_LR = 1e-3

def lr_scheduler(epoch, lr):
    return lr * 0.985

# callbacks = [
#     LearningRateScheduler(lr_scheduler, verbose=1)
# ]
def build_model(shape,n_class,n_latent = 64):
 
    inp = keras.Input(shape=shape, name="input")
    x = layers.Dense(128, activation="relu")(inp)
    latent = layers.Dense(n_latent, activation="relu")(x)
    dop = layers.Dropout(0.6)(latent)
    out = layers.Dense(n_class, activation="softmax")(dop)
    # out = layers.Dense(n_class, activation="sigmoid")(dop)


    clf = keras.Model(inputs=inp, outputs=out, name="Classifier")
    drt = keras.Model(inputs=inp, outputs=latent, name="DimensionalityReducer")

#     clf.summary()

    clf.compile(
    #     loss=keras.losses.BinaryCrossentropy(),
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer='adam',
        metrics=["accuracy"],
    )
    return clf,drt

def model_build(shape = (28, 28, 1), l1 = 1e-10):
    input_img = keras.Input(shape=shape)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    sh = x.shape[1:]
    print(sh)
    x = layers.Flatten()(x)

    encoded = layers.Dense(2, activation='relu',
                activity_regularizer=regularizers.l1(l1))(x)
    
    x = layers.Dense(np.prod(sh), activation='relu',
                activity_regularizer=regularizers.l1(l1))(encoded)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Reshape(sh)(x)
    
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
#     x = layers.UpSampling2D((2, 2))(x)
#     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)

    autoencoder = keras.Model(input_img, decoded)
    encoder = keras.Model(input_img, encoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder,encoder

# autoencoder,encoder = model_build()

# H = autoencoder.fit(x, x,
#                     epochs=100,
#                     batch_size=512,
#                     shuffle=True,
# #                     validation_data=(x_test, x_test)
#                    )


# inputs = keras.Input(shape=(original_dim,))
# h = layers.Dense(intermediate_dim, activation='relu')(inputs)

def build_model_dense(shape,n_class,n_latent = 64):
 
    inp = keras.Input(shape=shape, name="input")
    x = layers.Dense(128, activation="relu")(inp)
    latent = layers.Dense(n_latent, activation="relu")(x)
    dop = layers.Dropout(0.6)(latent)
    out = layers.Dense(n_class, activation="softmax")(dop)
    # out = layers.Dense(n_class, activation="sigmoid")(dop)


    clf = keras.Model(inputs=inp, outputs=out, name="Classifier")
    drt = keras.Model(inputs=inp, outputs=latent, name="DimensionalityReducer")

#     clf.summary()

    clf.compile(
    #     loss=keras.losses.BinaryCrossentropy(),
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer='adam',
        metrics=["accuracy"],
    )
    return clf,drt

def build_model_2dconv(shape,n_class,n_latent = 64,l1=1e-10):

    
    inp = keras.Input(shape=shape, name="input")

    x = layers.Conv2D(16, (2, 2), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(inp)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (2, 2), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    #     x = layers.MaxPooling2D((2, 2), padding='same')(x)
    #     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    sh = x.shape[1:]
#     print(sh)
    x = layers.Flatten()(x)

    latent = layers.Dense(n_latent, activation="relu")(x)
    dop = layers.Dropout(0.6)(latent)
    out = layers.Dense(n_class, activation="softmax",activity_regularizer=regularizers.l1(l1))(dop)

#     x = layers.Dense(2, activation='softmax',
#                 activity_regularizer=regularizers.l1(l1))(x)
    
    
    
    
    
#     inp = keras.Input(shape=shape, name="input")
#     x = layers.Dense(128, activation="relu")(inp)
#     latent = layers.Dense(n_latent, activation="relu")(x)
#     dop = layers.Dropout(0.6)(latent)
#     out = layers.Dense(n_class, activation="softmax")(dop)
#     # out = layers.Dense(n_class, activation="sigmoid")(dop)


    clf = keras.Model(inputs=inp, outputs=out, name="Classifier")
    drt = keras.Model(inputs=inp, outputs=latent, name="DimensionalityReducer")

#     clf.summary()

    clf.compile(
    #     loss=keras.losses.BinaryCrossentropy(),
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer='adam',
        metrics=["accuracy"],
    )
    return clf,drt


def add_class(clf,drt,n_class = None,loss=None,optimizer='adam',metrics=["accuracy"],summary=False):
    
    if n_class is None:
        n_class = clf.layers[-1].output_shape[1]+1
    if loss is None:
        loss = keras.losses.CategoricalCrossentropy()
        
    dclass = n_class-clf.layers[-1].output_shape[1]
    w,b = clf.layers[-1].get_weights()

    inp = keras.Input(shape=clf.layers[0].input_shape[0][1:], name="input")
    latent = drt(inp)
    dop = layers.Dropout(clf.layers[-2].rate)(latent)
    out = layers.Dense(n_class, activation="softmax")(dop)
    # out = layers.Dense(n_class, activation="sigmoid")(dop)

    clf2 = keras.Model(inputs=inp, outputs=out, name="Classifier2")

    #     clf.summary()

    w_new = np.concatenate([w,clf2.layers[-1].weights[0].numpy()[:,-dclass:]],axis=-1)
    b_new = np.concatenate([b,clf2.layers[-1].weights[1].numpy()[-dclass:]],axis=-1)

    clf2.layers[-1].set_weights([w_new,b_new])

    clf2.compile(
    #     loss=keras.losses.BinaryCrossentropy(),
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

    if summary:
        clf2.summary()
    return clf2


def stds_model(clf):
    ws = []
    bs = []
    for layer in clf.layers:
        if isinstance(layer,layers.Dense):
            ws.append(layer.weights[0].numpy().std())
            bs.append(layer.weights[1].numpy().std())
    return np.array(ws),np.array(bs)
# stds_model(clf)




def VAE(shape = (28, 28, 1),latent_dim = 2,l1 = 1e-10):
    
    original_dim = np.prod(shape[:-1])
    print(original_dim)
    input_img = keras.Input(shape=shape)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    #     x = layers.MaxPooling2D((2, 2), padding='same')(x)
    #     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    sh = x.shape[1:]
#     print(sh)
    x = layers.Flatten()(x)
    
    x = layers.Dense(2, activation='softmax',
                activity_regularizer=regularizers.l1(l1))(x)

#     latent_dim = np.prod(sh)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_sigma])

    # Create encoder
    encoder = keras.Model(input_img, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    # latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    # x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    # outputs = layers.Dense(original_dim, activation='sigmoid')(x)

    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')

    x = layers.Dense(np.prod(sh), activation='relu',
                activity_regularizer=regularizers.l1(l1))(latent_inputs)
    
    x = layers.Reshape(sh)(x)

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    #     x = layers.UpSampling2D((2, 2))(x)
    #     x = layers.Conv2D(8, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='relu', padding='same', activity_regularizer=regularizers.l1(l1))(x)

    decoder = keras.Model(latent_inputs, decoded, name='decoder')

    # instantiate VAE model
    decoded = decoder(encoder(input_img)[2])
    vae = keras.Model(input_img, decoded, name='vae_mlp')

#     reconstruction_loss = keras.losses.binary_crossentropy(input_img, decoded)
    reconstruction_loss = keras.losses.mse(input_img, decoded)
    reconstruction_loss *= original_dim
    
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    reconstruction_loss = K.mean(reconstruction_loss, axis=(-2,-1))
    
    vae_loss =  reconstruction_loss + kl_loss
    vae_loss = K.mean(vae_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return encoder,decoder,vae
