
feature_dim = 32
encoding_dim = 16
ae_epoch = 30
clf_epoch = 30
batch_size = 32

def encodeCategorical(df):
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()

    df['labels'] = le.fit_transform(df['labels'])
    df['protocol_type'] = le.fit_transform(df['protocol_type'])
    df['service'] = le.fit_transform(df['service'])
    df['flag'] = le.fit_transform(df['flag'])

    return df

def scaleData(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = sc.fit_transform(X_test)
    X_test = pd.DataFrame(X_test)

    return X_train, X_test

def reduceFeaturespace(X_train, X_test, y_train):
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    DTC = DecisionTreeClassifier()
    rfe = RFE(DTC, n_features_to_select= feature_dim).fit(X_train,y_train)
    indices = np.where(rfe.support_==True)[0]
    features = X_train.columns.values[indices]
    X_train, X_test = X_train[features], X_test[features]
    return X_train, X_test

def getdata():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils

    df = pd.read_csv('./datasets/NSLKDD/kdd_train.csv')
    df = encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    
    X_train, X_test = scaleData(X_train, X_test)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    X_train, X_test = reduceFeaturespace(X_train, X_test, y_train)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from models.autoencoders.binaryAE import BinaryAutoencoder
    from models.classifiers.CNN import CNNClassifier

    X_train, X_test, y_train, y_test = getdata()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    binary_ae = BinaryAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= 50, batch_size=32)
    binary_ae.train(X_train, X_test)
    binary_ae.freeze_encoder()
    encoder = binary_ae.encoder

    classifier = CNNClassifier(encoder= encoder,feature_dim= feature_dim)

