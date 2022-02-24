import numpy as np
np.random.seed(123)
import gzip
import _pickle as cPickle
import os
from collections import Counter

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KDTree

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.io import loadmat
import pandas as pd


WORD2VECPATH    = "../data/class_vectors.npy"
DATAPATH        = "../data/zeroshot_data.pkl"
MODELPATH       = "../model/"

def load_keras_model(model_path):
    with open(model_path +"model.json", 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"model.h5")
    return loaded_model

def save_keras_model(model, model_path):
    """save Keras model and its weights"""
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_json = model.to_json()
    with open(model_path + "model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_path + "model.h5")
    print("-> zsl model is saved.")
    return

def create_dataset():
    #导入橡胶球的数据，每个区域4*20
    data_xj=loadmat('E:/被动冲击/数据集/XJ/data.mat')
    data_xj=data_xj['dataset']

    #导入铁球的数据，每个区域4*10
    data_tg=loadmat('E:/被动冲击/数据集/TG/matlab.mat')
    data_tg=data_tg['dataset']
    
    #导入属性矩阵
    attribute_matrix_=pd.read_excel('E:/被动冲击/数据集/attribute.xlsx')
    attribute_matrix=attribute_matrix_.values
    #训练集和测试集的所有标签
    train_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    test_index=[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    train_index.sort()
    test_index.sort()

    #由于数据集中包含数据和标签，对其进行分离
    #train集合（橡胶球）
    #初始化
    trainlabel=[]
    trainattributelabel=[]
    traindata=[]
    #开始赋值
    traindata=data_xj[0:6400,:]
    traindata=traindata.T
    #选用传感器1的数据
    traindata_1=[]
    for i in list(range(traindata.shape[0])):
        if int(i%4)==0:
            traindata_1.append(traindata[i,:])
        
       
    for item in train_index:
        trainlabel += [item] * 20
        
        trainattributelabel+=[attribute_matrix[item,:]]*20
    #将list转化为array
    trainattributelabel=np.row_stack(trainattributelabel)
    trainlabel=np.row_stack(trainlabel)
    traindata=np.row_stack(traindata_1)
    #test集合(铁球)
    #初始化
    testlabel=[]
    testattributelabel=[]
    testdata=[]
    #开始赋值    
    testdata=data_tg[0:6400,:]
    testdata=testdata.T
    #选用传感器1的数据
    testdata_1=[]
    for i in list(range(testdata.shape[0])):
        if int(i%4)==0:
            testdata_1.append(testdata[i,:])
   
    for item in test_index:
        testattributelabel+=[attribute_matrix[item-16,:]]*10
        testlabel += [item] * 10
    #将list转化为array
    testattributelabel=np.row_stack(testattributelabel)    
    testlabel=np.row_stack(testlabel)
    testdata=np.row_stack(testdata_1)
    #将trainlabel转换为onehot,testlabel不转换
    y_train=to_categorical(trainlabel)
    y_zsl=testlabel
    y_zsl=np.array(y_zsl)
    x_train=traindata
    x_zsl=testdata
    #截取数据 数据长度6400 经过选取，选取[1500,1900]
    x_train=x_train[:,1500:1900]
    x_zsl=x_zsl[:,1500:1900]
    return (x_train, x_zsl), (y_train, y_zsl)


def custom_kernel_init(shape):
    class_vectors       = np.load(WORD2VECPATH,allow_pickle=True)
    training_vectors    = sorted([(label, vec) for (label, vec) in class_vectors if label in train_classes], key=lambda x: x[0])
    classnames, vectors = zip(*training_vectors)
    vectors             = np.asarray(vectors, dtype=np.float)
    vectors             = vectors.T
    
    attribute_matrix_=pd.read_excel('E:/被动冲击/数据集/attribute.xlsx')
    attribute_matrix=attribute_matrix_.values
    vectors=attribute_matrix[0:16,:]
    vectors=vectors.T
    return vectors

def  build_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(400,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NUM_ATTR, activation='relu'))
    model.add(Dense(NUM_CLASS, activation='softmax', trainable=False, kernel_initializer=custom_kernel_init))

    print("-> model building is completed.")
    return model


def train_model(model, train_data):
    x_train, y_train = train_data
    adam = Adam(lr=5e-5)
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = adam,
                  metrics   = ['categorical_accuracy', 'top_k_categorical_accuracy'])

    history = model.fit(x_train, y_train,
                        verbose         = 2,
                        epochs          = EPOCH,
                        batch_size      = BATCH_SIZE,
                        shuffle         = True)

    print("model training is completed.")
    return history

def main():

    global train_classes
    with open('train_classes.txt', 'r') as infile:
        train_classes = [str.strip(line) for line in infile]

    global zsl_classes
    with open('zsl_classes.txt', 'r') as infile:
        zsl_classes = [str.strip(line) for line in infile]

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # SET HYPERPARAMETERS

    global NUM_CLASS, NUM_ATTR, EPOCH, BATCH_SIZE
    NUM_CLASS = 16
    NUM_ATTR = 31
    BATCH_SIZE = 8
    EPOCH = 1000

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # TRAINING PHASE 对已知数据集（15个类别）进行训练

    (x_train,  x_zsl), (y_train, y_zsl) = create_dataset()
    model = build_model()
    train_model(model, (x_train, y_train))
    print(model.summary())

    

    
    # --------------------444-------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # CREATE AND SAVE ZSL MODEL

    inp         = model.input
    out         = model.layers[-2].output
    zsl_model   = Model(inp, out)
    print(zsl_model.summary())
    save_keras_model(zsl_model, model_path=MODELPATH)

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # EVALUATION OF ZERO-SHOT LEARNING PERFORMANCE
    #(x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl) = load_data()
    #zsl_model = load_keras_model(model_path=MODELPATH)

    attribute_matrix_=pd.read_excel('E:/被动冲击/数据集/attribute.xlsx')
    attribute_matrix=attribute_matrix_.values
    vectors=attribute_matrix
    
    #训练集和测试集的所有标签
    train_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    test_index=[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    train_index.sort()
    test_index.sort()
    classnames          = train_index+test_index
    #建立二叉搜索树，方便后面进行寻找最近邻
    tree        = KDTree(vectors)
    pred_zsl    = zsl_model.predict(x_zsl)

    top5, top3, top1 = 0, 0, 0
    for i, pred in enumerate(pred_zsl):
        pred            = np.expand_dims(pred, axis=0)
        dist_5, index_5 = tree.query(pred, k=5) #使用二叉搜索树进行查询最近的5个邻居
        pred_labels     = [classnames[index] for index in index_5[0]] #5个最近邻的labels name
        true_label      = y_zsl[i] #真实标签
        true_label=int(true_label)
        if true_label in pred_labels:
            top5 += 1
        if true_label in pred_labels[:3]:
            top3 += 1
        if true_label in [pred_labels[0]]:
            top1 += 1

    print()
    print("ZERO SHOT LEARNING SCORE")
    print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x_zsl))))
    print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x_zsl))))
    print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x_zsl))))
    return

if __name__ == '__main__':
    main()
