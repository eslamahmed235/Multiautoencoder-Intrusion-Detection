DEBUG = True
NUM_FOLDS = 3
feature_dim = 32
encoding_dim = 16
ae_epoch = 30
clf_epoch = 30
batch_size = 32

if DEBUG==True:
    NUM_FOLDS = 3
    ae_epoch = 1 
    clf_epoch = 1

def train_binary(x, y):
    from models.autoencoders.binaryAE import BinaryAutoencoder
    from models.classifiers.binaryClassifier import BinaryClassifier  
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    histories = []
    
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021) 
    for fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[test_idx]
        binary_ae = BinaryAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= ae_epoch, batch_size=32)
        binary_ae.train(X_train, X_valid)
        binary_ae.freeze_encoder()
        binary_encoder = binary_ae.encoder

        b_classifier = BinaryClassifier(encoder= binary_encoder,feature_dim= feature_dim, epochs= clf_epoch, batch_size=32)
        history = b_classifier.train(X_train, y_train, X_valid, y_valid)

        histories.append([history])

    enc_trainableParams = np.sum([np.prod(v.get_shape()) for v in binary_encoder.trainable_weights])
    enc_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in binary_encoder.non_trainable_weights])
    enc_totalParams = enc_trainableParams + enc_nonTrainableParams
    
    clf_trainableParams = np.sum([np.prod(v.get_shape()) for v in b_classifier.classifier.trainable_weights])
    clf_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in b_classifier.classifier.non_trainable_weights])
    clf_totalParams = clf_trainableParams + clf_nonTrainableParams
    
    totalParams = enc_totalParams + clf_totalParams
    
    return histories, totalParams

def train_multi(x, y):
    from models.autoencoders.multiAE import MultiAutoencoder
    from models.classifiers.multiclassClassifier import MulticlassClassifier
    from keras.utils import np_utils
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    ycat = np_utils.to_categorical(y)
    histories = []
    
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021) 
    for fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_valid = ycat[train_idx], ycat[test_idx]
        multi_ae = MultiAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= ae_epoch, batch_size=32)
        multi_ae.train(X_train, X_valid)
        multi_ae.freeze_encoder()
        multi_encoder = multi_ae.encoder

        multi_classifier = MulticlassClassifier(encoder= multi_encoder,feature_dim= feature_dim, num_classes = y_train.shape[1] ,epochs= clf_epoch, batch_size=32)
        history = multi_classifier.train(X_train, y_train, X_valid, y_valid)
        histories.append([history])

    enc_trainableParams = np.sum([np.prod(v.get_shape()) for v in multi_encoder.trainable_weights])
    enc_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in multi_encoder.non_trainable_weights])
    enc_totalParams = enc_trainableParams + enc_nonTrainableParams

    clf_trainableParams = np.sum([np.prod(v.get_shape()) for v in multi_classifier.classifier.trainable_weights])
    clf_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in multi_classifier.classifier.non_trainable_weights])
    clf_totalParams = clf_trainableParams + clf_nonTrainableParams
    
    totalParams = enc_totalParams + clf_totalParams

    return histories, totalParams


if __name__ == "__main__":
    
    from utils import getbinarydata, getattackdata, print_histories
    
    # X_train, X_test, y_train, y_test = getdata()
    
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    
    X_bin, y_bin = getbinarydata(feature_dim)
    
    print(X_bin.shape)
    print(y_bin.shape)

    bin_history, bin_params = train_binary(X_bin, y_bin)
 
    x_multi, y_multi = getattackdata(feature_dim)
    
    print(x_multi.shape)
    print(y_multi.shape)

    multi_history, multi_params = train_multi(x_multi, y_multi)

    print('\n\n\n')
    print('Printing Binary Classification histories')
    print('\nParams: ',bin_params)
    print_histories(bin_history)
    print('\n\n')
    print('Printing Multiclass Classification histories')
    print('\nParams: ',multi_params)
    print_histories(multi_history)


