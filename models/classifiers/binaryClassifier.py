class BinaryClassifier:
    def __init__(self, encoder, feature_dim, epochs, batch_size):
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.epochs = epochs
        self.batch_size = batch_size
    
    def build_model(self):

        import tensorflow.keras as k
        from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
        from tensorflow.keras.models import Model
        
        input_layer = Input(shape=(self.feature_dim, ))

        encoding = self.encoder(input_layer, training=False)

        layer1 = Dense(64, activation="relu")(encoding)
        layer1 = BatchNormalization()(layer1)
        layer1 = Dropout(0.2)(layer1)

        layer2 = Dense(64, activation="relu")(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = Dropout(0.3)(layer2)

        layer3 = Dense(128, activation="relu")(layer2)
        layer3 = BatchNormalization()(layer3)
        layer3 = Dropout(0.3)(layer3)

        layer4 = Dense(128, activation="relu")(layer3)
        layer4 = BatchNormalization()(layer4)
        layer4 = Dropout(0.2)(layer4)

        output_layer = Dense(1, activation="sigmoid")(layer4)

        classifier = Model(inputs=input_layer ,outputs=output_layer)
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'AUC'])
        classifier.summary()

        self.classifier = classifier

    def train(self, x_train, y_train, x_test, y_test):
        import tensorflow as tf

        self.build_model()

        def LRschedulerAE(epoch):
            import math
            initial_lrate = 0.01
            drop = 0.005
            epochs_drop = 5.0
            lrate = initial_lrate * math.pow(drop,  
                math.floor((1+epoch)/epochs_drop))
            return lrate

        clf_lr = tf.keras.callbacks.LearningRateScheduler(LRschedulerAE)

        history = self.classifier.fit(x_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=[clf_lr],
                    verbose=1).history
    def predict():
        pass

    