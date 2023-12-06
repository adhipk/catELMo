'''
Part of catELMo
(c) 2023 by  Pengfei Zhang, Michael Cai, Seojin Bang, Heewook Lee, and Arizona State University.
See LICENSE-CC-BY-NC-ND for licensing.
'''

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from numpy import mean, std
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MaxPooling2D,
    ReLU,
    GlobalMaxPooling1D,
    SeparableConv2D,
    MaxPooling1D
)
from tensorflow.keras.models import Model
from tensorflow.math import subtract
from tqdm import tqdm

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'


def get_inputs(embedding_type):
    # data_dir = "/mnt/disk07/user/pzhang84/data/tcr_repertoires_healthy_samples/combined_dataset_repTCRs"
    data_dir = 'datasets/embeddings'
    if embedding_type == 'catELMo':
        dat = pd.read_pickle(f"{data_dir}/catELMo_combined.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True)     
    elif embedding_type == 'blosum62':
        dat = pd.read_pickle(f"{data_dir}/BLOSUM62.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True)

    elif embedding_type == 'blosum62_22_24':
        dat = pd.read_pickle(f"{data_dir}/BLOSUM62_20_22.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True)

    elif embedding_type == 'SeqVec':
        dat = pd.read_pickle(f"{data_dir}/SeqVec.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 
    elif embedding_type == 'ProtBert':
        dat = pd.read_pickle(f"{data_dir}/ProtBert.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 
    elif embedding_type == 'catBert':
        dat = pd.read_pickle(f"{data_dir}/catBert.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 
    elif embedding_type == 'Doc2Vec':
        dat = pd.read_pickle(f"{data_dir}/Doc2Vec.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 
    elif embedding_type == 'catELMo_finetuned':
        dat = pd.read_pickle(f"{data_dir}/catELMo_finetuned.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True)  

    elif embedding_type == 'catBert_768_12_layers_mlm_nsp':
        dat = pd.read_pickle(f"{data_dir}/Small_Bert_mlm_nsp.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 

    elif embedding_type == 'catBert_768_12_layers_mlm':
        dat = pd.read_pickle(f"{data_dir}/Small_Bert_mlm.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 

    elif embedding_type == 'TCRbert':
        dat = pd.read_pickle(f"{data_dir}/TCRBert_mlm_12_layers.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 

    elif embedding_type == 'catBert_768_2_layers_mlm':
        dat = pd.read_pickle(f"{data_dir}/Small_Bert_mlm_2_layers.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 

    elif embedding_type == 'catELMo_4_layers_1024':
        dat = pd.read_pickle(f"{data_dir}/catELMo_4_layers_1024.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 

    elif embedding_type == 'catELMo_12_layers_1024':
        dat = pd.read_pickle(f"{data_dir}/Small_Bert_mlm_2_layers.pkl")
#         dat = dat.sample(frac=1).reset_index(drop=True) 

        
    return dat


def load_data_split(dat,split_type, seed):
    n_fold = 5
    idx_test_fold = 0
    idx_val_fold = -1
    idx_test = None
    idx_train = None
    x_pep = dat.epi
    x_tcr = dat.tcr
    
    if split_type == 'random':
        n_total = len(x_pep)
    elif split_type == 'epi':
        unique_peptides = np.unique(x_pep)
        n_total = len(unique_peptides)
    elif split_type == 'tcr':
        unique_tcrs = np.unique(x_tcr)
        n_total = len(unique_tcrs)
        
    np.random.seed(seed)    
    idx_shuffled = np.arange(n_total)
    np.random.shuffle(idx_shuffled)
    
    # Determine data split from folds
    n_test = int(round(n_total / n_fold))
    n_train = n_total - n_test

    # Determine position of current test fold
    test_fold_start_index = idx_test_fold * n_test
    test_fold_end_index = (idx_test_fold + 1) * n_test

    if split_type == 'random':
        # Split data evenly among evenly spaced folds
        # Determine if there is an outer testing fold
        if idx_val_fold < 0:
            idx_test = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove = idx_shuffled[test_fold_start_index:test_fold_end_index]
            idx_test = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            idx_train = list(set(idx_shuffled).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'epi':
        if idx_val_fold < 0:
            idx_test_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_pep = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_peptides = unique_peptides[idx_test_remove_pep]
            idx_test_pep = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_peptides = unique_peptides[idx_test_pep]
            idx_test = [index for index, pep in enumerate(x_pep) if pep in test_peptides]
            idx_test_remove = [index for index, pep in enumerate(x_pep) if pep in test_remove_peptides]
            idx_train = list(set(range(len(x_pep))).difference(set(idx_test)).difference(set(idx_test_remove)))
    elif split_type == 'tcr':
        if idx_val_fold < 0:
            idx_test_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)))
        else:
            validation_fold_start_index = args.idx_val_fold * n_test
            validation_fold_end_index = (args.idx_val_fold + 1) * n_test
            idx_test_remove_tcr = idx_shuffled[test_fold_start_index:test_fold_end_index]
            test_remove_tcrs = unique_tcrs[idx_test_remove_tcr]
            idx_test_tcr = idx_shuffled[validation_fold_start_index:validation_fold_end_index]
            test_tcrs = unique_tcrs[idx_test_tcr]
            idx_test = [index for index, tcr in enumerate(x_tcr) if tcr in test_tcrs]
            idx_test_remove = [index for index, tcr in enumerate(x_tcr) if tcr in test_remove_tcrs]
            idx_train = list(set(range(len(x_tcr))).difference(set(idx_test)).difference(set(idx_test_remove)))

    testData = dat.iloc[idx_test, :].sample(frac=1).reset_index(drop=True)
    trainData = dat.iloc[idx_train, :].sample(frac=1).reset_index(drop=True)
    

    print('================check Overlapping========================')
    print('number of overlapping tcrs: ', str(len(set(trainData.tcr).intersection(set(testData.tcr)))))
    print('number of overlapping epitopes: ', str(len(set(trainData.epi).intersection(set(testData.epi)))))
    
    # tcr_split testing read 
    X1_test_list, X2_test_list, y_test_list = testData.tcr_embeds.to_list(), testData.epi_embeds.to_list(),testData.binding.to_list()
    X1_test, X2_test, y_test = np.array(X1_test_list), np.array(X2_test_list), np.array(y_test_list)
    # tcr_split training read 
    X1_train_list, X2_train_list, y_train_list = trainData.tcr_embeds.to_list(), trainData.epi_embeds.to_list(),trainData.binding.to_list()
    X1_train, X2_train, y_train = np.array(X1_train_list), np.array(X2_train_list), np.array(y_train_list)
    return  X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData

# TCRConV Model
def create_cnn3ab_model(input_a,input_b, filters=[120, 100, 80, 60], kernel_sizes=[5, 9, 15, 21, 3], dos=[0.1, 0.2], pool='max'):

    # CNN A part
    cnn1a = Conv1D(filters[0], kernel_sizes[0], padding='same', activation='relu')(input_a)
    cnn2a = Conv1D(filters[1], kernel_sizes[1], padding='same', activation='relu')(input_a)
    cnn3a = Conv1D(filters[2], kernel_sizes[2], padding='same', activation='relu')(input_a)
    cnn4a = Conv1D(filters[3], kernel_sizes[3], padding='same', activation='relu')(input_a)

    merged_cnn_a = concatenate([cnn1a, cnn2a, cnn3a, cnn4a], axis=-1)
    merged_cnn_a = BatchNormalization()(merged_cnn_a)
    merged_cnn_a = Dropout(dos[0])(merged_cnn_a)

    cnn5a = Conv1D(100, kernel_sizes[4], padding='same', activation='relu')(merged_cnn_a)
    cnn5a = BatchNormalization()(cnn5a)
    cnn5a = MaxPooling1D(pool_size=2)(cnn5a)
    cnn5a = Flatten()(cnn5a)

    # Parallel Linear Neural Network (LNN) A part
    dense_ia = Dense(256, activation='relu')(Flatten()(input_a))

    # CNN B part
    cnn1b = Conv1D(filters[0], kernel_sizes[0], padding='same', activation='relu')(input_b)
    cnn2b = Conv1D(filters[1], kernel_sizes[1], padding='same', activation='relu')(input_b)
    cnn3b = Conv1D(filters[2], kernel_sizes[2], padding='same', activation='relu')(input_b)
    cnn4b = Conv1D(filters[3], kernel_sizes[3], padding='same', activation='relu')(input_b)

    merged_cnn_b = concatenate([cnn1b, cnn2b, cnn3b, cnn4b], axis=-1)
    merged_cnn_b = BatchNormalization()(merged_cnn_b)
    merged_cnn_b = Dropout(dos[0])(merged_cnn_b)

    cnn5b = Conv1D(100, kernel_sizes[4], padding='same', activation='relu')(merged_cnn_b)
    cnn5b = BatchNormalization()(cnn5b)
    cnn5b = MaxPooling1D(pool_size=2)(cnn5b)
    cnn5b = Flatten()(cnn5b)

    # Parallel Linear Neural Network (LNN) B part
    dense_ib = Dense(256, activation='relu')(Flatten()(input_b))

    # Combined part
    merged = concatenate([cnn5a, dense_ia, cnn5b, dense_ib], axis=-1)
    merged = Dropout(dos[1])(merged)
    output_layer = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[input_a, input_b], outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

def catelmo_model(inputA,inputB):
    
    x = Dense(2048,kernel_initializer = 'he_uniform')(inputA)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = tf.nn.silu(x)
    x = Model(inputs=inputA, outputs=x)
    
    y = Dense(2048,kernel_initializer = 'he_uniform')(inputB)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = tf.nn.silu(y)
    y = Model(inputs=inputB, outputs=y)
#     combined = concatenate([x.output, y.output, abs(subtract(x.output,y.output))])
    combined = concatenate([x.output, y.output])
    
    z = Dense(1024)(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = tf.nn.silu(z)
    z = Dense(1, activation='sigmoid')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    model.summary()
    return model
    
def train_(embedding_name,X1_train, X2_train, y_train, X1_test, X2_test, y_test,useCNN=False):
    # define two sets of inputs
    inputA = Input(shape=(len(X1_train[0]),))
    inputB = Input(shape=(len(X2_train[0]),))
    if useCNN:
        model = create_cnn3ab_model(inputA,inputB)
        model_name = 'TCRConv'
    else:
        model = catelmo_model(inputA,inputB)
        model_name = 'catELMo_4_layers_1024'
    ## model fit
    checkpoint_filepath = f'models/{model_name}/' + embedding_name +  '.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 30)
    model.fit([X1_train,X2_train], y_train, verbose=0, validation_split=0.20, epochs=200, batch_size = 32, callbacks=[es, model_checkpoint_callback])
    # model.save('models/' + model_name + embedding_name + '.hdf5')
    yhat = model.predict([X1_test, X2_test])
    
    print('================Performance========================')
    print(embedding_name+'AUC: ' + str(roc_auc_score(y_test, yhat)))

    
    yhat[yhat>=0.5] = 1
    yhat[yhat<0.5] = 0
    
    accuracy = accuracy_score(y_test, yhat)
    precision1 = precision_score(
        y_test, yhat, pos_label=1, zero_division=0)
    precision0 = precision_score(
        y_test, yhat, pos_label=0, zero_division=0)
    recall1 = recall_score(y_test, yhat, pos_label=1, zero_division=0)
    recall0 = recall_score(y_test, yhat, pos_label=0, zero_division=0)
    f1macro = f1_score(y_test, yhat, average='macro')
    f1micro = f1_score(y_test, yhat, average='micro')
    
    print('precision_recall_fscore_macro ' + str(precision_recall_fscore_support(y_test,yhat, average='macro')))
    print('acc is '  + str(accuracy))
    print('precision1 is '  + str(precision1))
    print('precision0 is '  + str(precision0))
    print('recall1 is '  + str(recall1))
    print('recall0 is '  + str(recall0))
    print('f1macro is '  + str(f1macro))
    print('f1micro is '  + str(f1micro))

    
def main(embedding, split,fraction,seed, gpu,use_cnn):
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    dat = get_inputs(embedding)
    tr_dat = dat
    tr_dat = dat.sample(frac=fraction, replace=True, random_state=seed).reset_index(drop=True) # comment this out if no fraction used
    X1_train, X2_train, y_train, X1_test, X2_test, y_test, testData, trainData = load_data_split(tr_dat,split, seed)
    train_(embedding + '_' + split + '_seed_' + str(seed) + '_fraction_' + str(fraction), X1_train, X2_train, y_train, X1_test, X2_test, y_test,useCNN=use_cnn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str,help='elmo or blosum62')
    parser.add_argument('--split', type=str,help='random, tcr or epi')
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--fraction', type=float, default=1.0) 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_cnn',action='store_true', default=False)
    args = parser.parse_args()
    main(args.embedding, args.split, args.fraction, args.seed, args.gpu.args.use_cnn)
