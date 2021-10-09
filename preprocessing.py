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