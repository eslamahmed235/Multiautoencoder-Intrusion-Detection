class BinaryAutoencoder:
    def __init__(self, inp_dim, enc_dim, epochs, batch_size):
        self.input_dim = inp_dim
        self.encoding_dim = enc_dim 
        self.epochs = epochs
        self.batch_size = batch_size

    def build_model(self):
        import tensorflow.keras as k
        from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
        from tensorflow.keras.models import Model
        
        ae_input_layer = Input(shape=(self.input_dim, ))

        enc = Dense(32, activation="swish")(ae_input_layer)
        enc = BatchNormalization()(enc)
        enc = Dense(self.encoding_dim, activation="swish")(enc)

        dec = BatchNormalization()(enc)
        dec = Dense(32, activation="swish")(dec)
        dec = BatchNormalization()(enc)
        dec = Dense(self.input_dim, activation="swish")(dec)

        autoencoder = Model(inputs=ae_input_layer, outputs=dec)
        encoder = Model(inputs=ae_input_layer, outputs=enc)
        autoencoder.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])

        self.autoencoder = autoencoder
        self.encoder = encoder

    def LRschedulerAE(epoch):
        initial_lrate = 0.01
        drop = 0.005
        epochs_drop = 5.0
        lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
        return lrate

    def train(self, x_train, x_test):
        import tensorflow as tf

        self.build_model()

        ae_lr = tf.keras.callbacks.LearningRateScheduler(self.LRschedulerAE)

        history = self.autoencoder.fit(x_train, x_train, 
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks = [ae_lr],
                    verbose=1).history

    def evaluate():
        pass

    def get_encoder(self):
        return self.encoder

    