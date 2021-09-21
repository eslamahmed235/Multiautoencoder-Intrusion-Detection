
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

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = sc.fit_transform(X_test)
    X_test = pd.DataFrame(X_test)

    return X_train, X_test

def reduceFeaturespace():
    pass

def getdata():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils

    df = pd.read_csv('../input/nslkdd/kdd_train.csv')
    df = encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    X_train, X_test = scaleData(X_train, X_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pass