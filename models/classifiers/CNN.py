class CNNClassifier:
    def __init__(self):
        pass
    
    def build_model(self):

        import tensorflow.keras as k
        from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
        from tensorflow.keras.models import Model
        
        input_layer = Input(shape=(feature_dim, ))

        encoding = encoder(input_layer, training=False)

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

        output_layer = Dense(23, activation="softmax")(layer4)

        classifier = Model(inputs=input_layer ,outputs=output_layer)

    def train():
        pass

    def predict():
        pass

    