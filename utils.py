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

def getbinarydata(feature_dim):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace

    df = pd.read_csv('./datasets/NSLKDD/KDDTrain+.txt')
    cols = get_cols()
    df.columns = cols
    print(df.labels.unique())
    df['is_attacked'] = df.apply(lambda x: 0 if x['labels']=='normal' else 1, axis=1)
    
    df= encodeCategorical(df)
    
    x = df.drop('labels', axis=1)
    x = x.drop('is_attacked', axis=1)
    x = x.drop('level', axis=1)
    y = df.loc[:, ['is_attacked']]
    
    x = scaleData(x)

    x = reduceFeaturespace(x, y, feature_dim, 'dtc')

    return x, y

def getattackdata(feature_dim, category):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils
    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace
    
    df = pd.read_csv('./datasets/NSLKDD/KDDTrain+.txt')
    cols = get_cols()
    df.columns = cols
    print(df.columns)
    categories = get_labels(category)
    df['drop'] = df.apply(lambda x: 0 if x['labels'] == (l for l in categories) else 1)
    idx = np.where(df['drop']==1)[0]
    df = df.drop(idx)
    
    
    # idx = np.where(df['labels'] != x for x in labels)[0]
    # df = df.drop(idx)

    df= encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]
    
    x = scaleData(x)
    # y = np_utils.to_categorical(y)
    
    x = reduceFeaturespace(x, y, feature_dim, 'dtc')

    return x, y

def print_histories(histories):
    for i in range(len(histories)):
        print('-'*15, '>', f'Fold {i+1}', '<', '-'*15)
        print(histories[i])


def get_labels(attack_class):
    dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
    probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
    privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
    access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']
    if attack_class == 'ddos':
        return dos_attacks

def get_cols():
    columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes'
    ,'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised'
    ,'root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files'
    ,'num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
    ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate'
    ,'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate'
    ,'dst_host_rerror_rate','dst_host_srv_rerror_rate','labels','level'])
    
    return columns