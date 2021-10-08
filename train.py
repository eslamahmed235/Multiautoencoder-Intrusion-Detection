NUM_FOLDS = 3
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

def scaleData(x):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    sc = StandardScaler()
    x = sc.fit_transform(x)
    x = pd.DataFrame(x)

    return x

def reduceFeaturespace(X_train, y_train):
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    DTC = DecisionTreeClassifier()
    rfe = RFE(DTC, n_features_to_select= feature_dim).fit(X_train,y_train)
    indices = np.where(rfe.support_==True)[0]
    features = X_train.columns.values[indices]
    X_train= X_train[features]
    return X_train

def getdata():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils

    df = pd.read_csv('./datasets/NSLKDD/kdd_train.csv')
    df= encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    
    X_train, X_test = scaleData(X_train, X_test)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # X_train, X_test = reduceFeaturespace(X_train, X_test, y_train)

    return X_train, X_test, y_train, y_test

def getbinarydata():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('./datasets/NSLKDD/kdd_train.csv')
    
    df['is_attacked'] = df.apply(lambda x: 0 if x['labels']=='normal' else 1, axis=1)
    
    df= encodeCategorical(df)
    
    x = df.drop('labels', axis=1)
    x = x.drop('is_attacked', axis=1)
    y = df.loc[:, ['is_attacked']]
    
    x = scaleData(x)

    x = reduceFeaturespace(x, y)

    return x, y

def getattackdata():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils

    df = pd.read_csv('./datasets/NSLKDD/kdd_train.csv')
    idx = np.where(df['labels']=='normal')[0]
    df = df.drop(idx)
    df= encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    
    X_train, X_test = scaleData(X_train, X_test)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    X_train, X_test = reduceFeaturespace(X_train, X_test, y_train)

    return X_train, X_test, y_train, y_test

def train_binary(x, y):
    from models.autoencoders.binaryAE import BinaryAutoencoder
    from models.classifiers.binaryClassifier import BinaryClassifier

    from sklearn.model_selection import StratifiedKFold
    
    histories = []
    
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021) 
    for fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[test_idx]
        binary_ae = BinaryAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= 1, batch_size=32)
        binary_ae.train(X_train, X_valid)
        binary_ae.freeze_encoder()
        binary_encoder = binary_ae.encoder

        b_classifier = BinaryClassifier(encoder= binary_encoder,feature_dim= feature_dim, epochs= 1, batch_size=32)
        history = b_classifier.train(X_train, y_train, X_valid, y_valid)
        histories.append([history])
    
    print('Printing histories')

    for i in range(len(histories)):
        print('-'*15, '>', f'Fold {i+1}', '<', '-'*15)
        print(histories[i])

if __name__ == "__main__":
    
    # from models.autoencoders.multiAE import MultiAutoencoder
    
    # from models.classifiers.multiclassClassifier import MulticlassClassifier

    # X_train, X_test, y_train, y_test = getdata()
    
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    
    X_bin, y_bin = getbinarydata()
    
    print(X_bin.shape)
    print(y_bin.shape)

    train_binary(X_bin, y_bin)
 
    # X_train_multi, X_test_multi, y_train_multi, y_test_multi = getattackdata()
    
    # print(X_train_multi.shape)
    # print(X_test_multi.shape)
    # print(y_train_multi.shape)
    # print(y_test_multi.shape)

    # multi_ae = MultiAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= 10, batch_size=32)
    # multi_ae.train(X_train_multi, X_test_multi)
    # multi_ae.freeze_encoder()
    # multi_encoder = multi_ae.encoder

    # multi_classifier = MulticlassClassifier(encoder= multi_encoder,feature_dim= feature_dim, num_classes = y_test_multi.shape[1] ,epochs= 20, batch_size=32)
    # multi_classifier.train(X_train_multi, y_train_multi, X_test_multi, y_test_multi)

    # classifier = CNNClassifier(encoder= encoder,feature_dim= feature_dim, epochs= 20, batch_size=32)
    # classifier.train(X_train, y_train, X_test, y_test)

