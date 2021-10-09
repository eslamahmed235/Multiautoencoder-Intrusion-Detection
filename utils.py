def getdata():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils

    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace

    df = pd.read_csv('./datasets/NSLKDD/kdd_train.csv')
    df= encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    
    X_train, X_test = scaleData(X_train, X_test)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    X_train, X_test = reduceFeaturespace(X_train, X_test, y_train)

    return X_train, X_test, y_train, y_test

def getbinarydata():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace

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
    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace
    
    df = pd.read_csv('./datasets/NSLKDD/kdd_train.csv')
    idx = np.where(df['labels']=='normal')[0]
    df = df.drop(idx)
    df= encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]
    
    x = scaleData(x)
    # y = np_utils.to_categorical(y)
    
    x = reduceFeaturespace(x, y)

    return x, y

def print_histories(histories):
    for i in range(len(histories)):
        print('-'*15, '>', f'Fold {i+1}', '<', '-'*15)
        print(histories[i])