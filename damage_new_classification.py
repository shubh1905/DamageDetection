#!/usr/bin/env python
# coding: utf-8

# In[30]:


import argparse
import os, shutil
from keras.models import load_model,Model
import tensorflow_addons as tfa
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU,Conv2D, Input,Dense,MaxPooling2D,Flatten,concatenate,subtract,Dropout,add
import tensorflow as tf
import pandas as pd
from keras.callbacks.callbacks import CSVLogger,ModelCheckpoint
import keras.backend as K
K.clear_session()
from keras.utils import multi_gpu_model
import os
np.random.seed(47)
tf.random.set_seed(2)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

original_dataset_dir =None

#original_dataset_dir = '/home/shubham/mlp_proj/xbd'

def recall_1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
       
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision= precision(y_true, y_pred)
    recall= recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def generator_siamese(generator,df,batch_size,img_height,img_width,direc):
    #df = df.replace({"labels" : damage_intensity_encoding })
    gen_1=generator.flow_from_dataframe(class_mode="categorical",
                                        batch_size=batch_size,
                                        dataframe=df,
                                        directory=direc,
                                        x_col="uuid_pre",
                                        y_col="labels",
                                        target_size=(img_height,img_width),
                                        seed=47,
                                        shuffle=False,validate_filenames=False)
    gen_2=generator.flow_from_dataframe(class_mode="categorical",
                                        batch_size=batch_size,
                                        dataframe=df,
                                        directory=direc,
                                        x_col="uuid_post",
                                        y_col="labels",
                                        target_size=(img_height,img_width),
                                        seed=47,
                                        shuffle=False,validate_filenames=False)
    while True:
        x1=gen_1.next()
        x2=gen_2.next()
        yield [x1[0],x2[0]],x2[1]

def generator_normal(generator,df,batch_size,img_height,img_width,direc):
    #df = df.replace({"labels" : damage_intensity_encoding })
    gen_1=generator.flow_from_dataframe(class_mode="categorical",
                                        batch_size=batch_size,
                                        dataframe=df,
                                        directory=direc,
                                        x_col="uuid_post",
                                        y_col="labels",
                                        target_size=(img_height,img_width),
                                        seed=47,
                                        shuffle=False)
    return gen_1
        
def generate_csv(dataframe):
    def add_post(x,string):
        tem=x.split(".")
        return tem[0]+string+".png"
    temp=dataframe
    store_pre=temp["uuid"].transform(lambda x:add_post(x,"_pre_"))
    store_post=temp["uuid"].transform(lambda x:add_post(x,"_post_"))
    store_labels=temp["labels"].transform(lambda x:str(x))
    temp["uuid_post"]=store_post
    temp["uuid_pre"]=store_pre
    temp["labels"]=store_labels
    return temp

def generate_xBD_baseline_model(dire):
    #weights = dire+'/resnet50_weight.h5'
    inputs = Input(shape=(128, 128, 3))
    base_model = ResNet50(include_top=False, weights=dire+'/resnet50_weight.h5')
    for layer in base_model.layers:
        layer.trainable = False

    x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(128, 128, 3))(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

    x = Flatten()(x)

    base_resnet = base_model(inputs)
    base_resnet = Flatten()(base_resnet)

    concated_layers = concatenate([x, base_resnet])

    concated_layers = Dense(2024, activation='relu')(concated_layers)
    concated_layers = Dense(524, activation='relu')(concated_layers)
    concated_layers = Dense(124, activation='relu')(concated_layers)
    output = Dense(4, activation='softmax')(concated_layers)

    model = Model(inputs=inputs, outputs=output)
    return model

def generate_small_cnn():
    inputs = Input(shape=(128, 128, 3))
    x = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(128, 128, 3))(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(x)

    x = Flatten()(x)
    concated_layers = Dense(2024, activation='relu')(x)
    concated_layers = Dense(524, activation='relu')(concated_layers)
    concated_layers = Dense(124, activation='relu')(concated_layers)
    output = Dense(2, activation='softmax')(concated_layers)

    model = Model(inputs=inputs, outputs=output)
    return model

def generate_small_siamese_cnn():
    input_imga = Input(shape=(128, 128, 3))
    input_imgb = Input(shape=(128, 128, 3))
    
    conv_1aa=Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')
    conv_1ab=Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')
    
    mxpool_1aa=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    mxpool_1ab=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    
    conv_2a=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    conv_2ab=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    
    mxpool_2a=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    mxpool_2ab=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    
    conv_3a=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    conv_3ab=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    
    mxpool_3a= MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    mxpool_3ab= MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    
    flata=Flatten()
    flatb=Flatten()   
    
    
    dense_1 = Dense(2024, activation='relu')
    dense_2= Dense(524, activation='relu')
    dense_3 = Dense(124, activation='relu')
    output = Dense(4, activation='softmax')
    
    
    layer_1a=conv_1aa(input_imga)
    layer_1b=conv_1ab(input_imgb)
    
    layer_2a=mxpool_1aa(layer_1a)
    layer_2b=mxpool_1ab(layer_1b)
    
    layer_3a=conv_2a(layer_2a)
    layer_3b=conv_2ab(layer_2b)
    
    layer_4a=mxpool_2a(layer_3a)
    layer_4b=mxpool_2ab(layer_3b)
    
    layer_5a=conv_3a(layer_4a)
    layer_5b=conv_3ab(layer_4b)
    
    layer_6a=mxpool_3a(layer_5a)
    layer_6b=mxpool_3ab(layer_5b)
    
    layer_8a=flata(layer_6a)
    layer_8b=flatb(layer_6b)
    
    concat=add([layer_8a,layer_8b])
    concat=Dropout(0.60)(concat)
    concat=dense_1(concat)
    concat=Dropout(0.60)(concat)
    concat=dense_2(concat)
    #concat=Dropout(0.5)(concat)
    concat=dense_3(concat)
    #concat=Dropout(0.5)(concat)
    concat=output(concat)
    model=Model(inputs=[input_imga,input_imgb],outputs=concat)

    return model

def generate_small_weight_siamese_cnn():
    input_imga = Input(shape=(128, 128, 3))
    input_imgb = Input(shape=(128, 128, 3))
    dense_2= Dense(524, activation='relu')
    dense_3 = Dense(124, activation='relu')
    output = Dense(4, activation='softmax')
    conv_1aa=Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')
    #conv_1ab=Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')
    
    mxpool_1aa=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    #mxpool_1ab=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    
    conv_2a=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    #conv_2ab=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    
    mxpool_2a=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    #mxpool_2ab=MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    
    conv_3a=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    #conv_3ab=Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
    
    mxpool_3a= MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    #mxpool_3ab= MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    
    flata=Flatten()
    #flatb=Flatten()   
    
    
    dense_1 = Dense(2024, activation='relu')
    #dense_2= Dense(524, activation='relu')
    #dense_3 = Dense(124, activation='relu')
    #output = Dense(4, activation='softmax')
    
    def gen(input_img):
        
        
        
        layer_1a=conv_1aa(input_img)
        #layer_1b=conv_1aa(input_imgb)
        
        layer_2a=mxpool_1aa(layer_1a)
        #layer_2b=mxpool_1aa(layer_1b)
        
        layer_3a=conv_2a(layer_2a)
        #layer_3b=conv_2a(layer_2b)
        
        layer_4a=mxpool_2a(layer_3a)
        #layer_4b=mxpool_2a(layer_3b)
        
        layer_5a=conv_3a(layer_4a)
        #layer_5b=conv_3a(layer_4b)
        
        layer_6a=mxpool_3a(layer_5a)
        #layer_6b=mxpool_3a(layer_5b)
        
        layer_8a=flata(layer_6a)
        #layer_8b=flatb(layer_6b)
        
        return layer_8a

    layer_8a=gen(input_imga)
    layer_8b=gen(input_imgb)
    
    concat=subtract([layer_8a,layer_8b])
    concat=Dropout(0.60)(concat)
    concat=dense_1(concat)
    concat=Dropout(0.60)(concat)
    concat=dense_2(concat)
    #concat=Dropout(0.5)(concat)
    concat=dense_3(concat)
    #concat=Dropout(0.5)(concat)
    concat=output(concat)
    model=Model(inputs=[input_imga,input_imgb],outputs=concat)

    return model

def generate_siamese_model():
    input_imga = Input(shape=(128, 128, 3))
    input_imgb = Input(shape=(128, 128, 3))
    
    conv_1aa=Conv2D(32,(3,3), activation = 'relu')
    conv_1ab=Conv2D(32,(3,3), activation = 'relu')
    
    mxpool_1aa=MaxPooling2D((2,2))
    mxpool_1ab=MaxPooling2D((2,2))
    
    conv_2a=Conv2D(64,(3,3), activation = 'relu')
    conv_2ab=Conv2D(64,(3,3), activation = 'relu')
    
    mxpool_2a=MaxPooling2D((2,2))
    mxpool_2ab=MaxPooling2D((2,2))
    
    conv_3a=Conv2D(128,(3,3), activation = 'relu')
    conv_3ab=Conv2D(128,(3,3), activation = 'relu')
    
    mxpool_3a=MaxPooling2D((2,2))
    mxpool_3ab=MaxPooling2D((2,2))
    
    conv_4a=Conv2D(128,(3,3), activation = 'relu')
    conv_4ab=Conv2D(128,(3,3), activation = 'relu')
    
    mxpool_4a=MaxPooling2D((2,2))       
    mxpool_4ab=MaxPooling2D((2,2))

    flata=Flatten()
    flatb=Flatten()


    layer_1a=conv_1aa(input_imga)
    layer_1b=conv_1ab(input_imgb)
    
    layer_2a=mxpool_1aa(layer_1a)
    layer_2b=mxpool_1ab(layer_1b)
    
    layer_3a=conv_2a(layer_2a)
    layer_3b=conv_2ab(layer_2b)
    
    layer_4a=mxpool_2a(layer_3a)
    layer_4b=mxpool_2ab(layer_3b)
    
    layer_5a=conv_3a(layer_4a)
    layer_5b=conv_3ab(layer_4b)
    
    layer_6a=mxpool_3a(layer_5a)
    layer_6b=mxpool_3ab(layer_5b)
    
    layer_8a=flata(layer_6a)
    layer_8b=flatb(layer_6b)
    
    concat=subtract([layer_8a,layer_8b])
    dropout=Dropout(0.5)(concat)
    dense_1=Dense(1000,activation = 'relu')(dropout)
    dropout=Dropout(0.5)(concat)
    dense_1=Dense(512,activation = 'relu')(dropout)
    dense_2=Dense(4, activation = 'softmax')(dense_1)
    model=Model(inputs=[input_imga,input_imgb],outputs=dense_2)
    return model

def model_selector(model_name,experiment_name,augment,augment_level,batch_size,original_dataset_dir):

 
    run_directory=original_dataset_dir+"/"+experiment_name
    model=None
    if model_name=="base":
        model=generate_xBD_baseline_model(original_dataset_dir)
    if model_name=="small_cnn":
        model=generate_small_cnn()
    if model_name=="small_siamese_cnn":
        model=generate_small_siamese_cnn()
    if model_name=="siamese":
        model=generate_siamese_model()
    if model_name=="siamese_2":
        model=generate_small_weight_siamese_cnn()
    if model_name=="":
        model=None
    if model==None:
        return "Error!!!!"
    print(model.summary())
    train_datagen=None
    if augment==True:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                        rotation_range = 40,
                        width_shift_range = 0.2,
                        height_shift_range = 0.2,
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        horizontal_flip = True)
    else:
        train_datagen= ImageDataGenerator(rescale=1./255)
        
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    
    #model=multi_gpu_model(model, gpus=1)
    try:
        model = multi_gpu_model(model)
    except:
        pass
    f1=tfa.metrics.F1Score(num_classes=4,average="micro")
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(), metrics = ['acc',f1_score,precision_1,recall_1])
    csv_callback=CSVLogger(run_directory+"/log.csv", separator=',', append=False)
    chk_callback=ModelCheckpoint(run_directory+"/model", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    
    train_csv=pd.read_csv(original_dataset_dir+"/polygon_csv/new_all_balanced_train.csv")
    valid_csv=pd.read_csv(original_dataset_dir+"/polygon_csv/new_all_balance_valid.csv")
    test_csv=pd.read_csv(original_dataset_dir+"/polygon_csv/new_all_balanced_test.csv")
    
    df_train=generate_csv(train_csv)
    df_valid=generate_csv(valid_csv)
    df_test=generate_csv(test_csv)
    #print(df_valid["labels"])
    #df_train=df_train[(df_train["labels"]==str(0) )| (df_train["labels"]==str(3))]
    #df_valid=df_valid[(df_valid["labels"]==str(0))|(df_valid["labels"]==str(3))]
    #df_test=df_test[(df_test["labels"]==str(0))|(df_test["labels"]==str(3))]
  
    print("\n\n\n\n\n")
    print(df_train.shape)
    train_generator=None
    valid_generator=None
    test_generator=None
    full_test_generator=None
    if model_name=="siamese" or model_name=="small_siamese_cnn" or model_name=="siamese_2":
        train_generator=generator_siamese(train_datagen,df_train,batch_size,128,128,original_dataset_dir+"/balanced_data/")
        valid_generator=generator_siamese(test_datagen,df_valid,batch_size,128,128,original_dataset_dir+"/balanced_data/")
        test_generator=generator_siamese(test_datagen,df_test,batch_size,128,128,original_dataset_dir+"/balanced_data")
        
        print("\n\nsiamese generator\n\n")
        
        history = model.fit(
            train_generator,
            steps_per_epoch=df_train.shape[0]//(batch_size*1)*augment_level,
            epochs=100,
            validation_data=valid_generator,
            validation_steps=df_valid.shape[0]//(batch_size),
            callbacks=[csv_callback,chk_callback],verbose=1)
        
        evaluation=model.evaluate(test_generator,
                              steps=df_test.shape[0]//(batch_size*1),verbose=1
                              )
        print(dict(zip(model.metrics_names, evaluation)))
    if model_name=="base" or model_name=="small_cnn":
        history = model.fit(
            train_datagen.flow_from_dataframe(class_mode="categorical",
                                        batch_size=batch_size,
                                        dataframe=df_train,
                                        directory=original_dataset_dir+"/balanced_data/",
                                        x_col="uuid_post",
                                        y_col="labels",
                                        target_size=(128,128),
                                        seed=47,
                                        shuffle=False,validate_filenames=False),
            steps_per_epoch=df_train.shape[0]//(batch_size*1)*augment_level,
            epochs=100,
            validation_data=test_datagen.flow_from_dataframe(class_mode="categorical",
                                        batch_size=batch_size,
                                        dataframe=df_valid,
                                        directory=original_dataset_dir+"/balanced_data/",
                                        x_col="uuid_post",
                                        y_col="labels",
                                        target_size=(128,128),
                                        seed=47,
                                        shuffle=False,validate_filenames=False),
            validation_steps=df_valid.shape[0]//batch_size,
            callbacks=[csv_callback,chk_callback],
            verbose=1)
        
        csv_callback_evaluate=CSVLogger(run_directory+"/log.csv", separator=',', append=False)
        evaluation=model.evaluate(test_datagen.flow_from_dataframe(class_mode="categorical",
                                        batch_size=batch_size,
                                        dataframe=df_test,
                                        directory=original_dataset_dir+"/balanced_data/",
                                        x_col="uuid_post",
                                        y_col="labels",
                                        target_size=(128,128),
                                        seed=47,
                                        shuffle=False),
                                        steps=df_test.shape[0]//(batch_size*1),
                                  verbose=1
                                        )
    f=open(run_directory+"/test.txt")
    f.write(dict(zip(model.metrics_names, evaluation)))
    f.close()
        
    
    
    
    
    
    return "Program conpleted"


# In[28]:


#model_selector(model_name="small_siamese_cnn",batch_size=20)


# In[31]:


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--data',
                        required=True,
                        metavar="/path/to/xBD",
                        help="Full path to the train data directory")

    parser.add_argument('--model_name',
                        required=True,
                        help="Name of model")
    parser.add_argument('--experiment_name',
                        required=True,
                        help="Experiment Name")
    parser.add_argument('--augment',
                        default=False,
                        help="Augment or not")
    parser.add_argument('--augment_level',
                        default=1,
                        help="Number of times to augment")

    parser.add_argument('--batch_size',
                        required=True,
                        help="Batch Size")
    args = parser.parse_args()
    original_dataset_dir = args.data
    model_selector(model_name=args.model_name,
                   experiment_name=args.experiment_name,
                   augment=bool(args.augment),
                   augment_level=int(args.augment_level),
                   batch_size=int(args.batch_size),
                   original_dataset_dir=args.data)
    
if __name__ == '__main__':
    main()

