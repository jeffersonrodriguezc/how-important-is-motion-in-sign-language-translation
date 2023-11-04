import keras
import tensorflow
import math
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import os
import pickle
import numpy as np

def learningSigns(model, training_generator, validation_generator, args):
    
    def step_decay(epoch):
        initial_lrate = args.lr
        drop = args.decay
        epochs_drop = args.nIters
        lrate = initial_lrate * math.pow(drop,  
               math.floor((1+epoch)/epochs_drop))
        return lrate
    
    
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
    
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
    
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """
    
        weights = K.variable(weights)
        
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
    
        return loss
    
    # Loading the dictionary
    word2int = open(os.getcwd()+'/results/dataTrain_phoenix_sentences/dicts/word2int.pkl',"rb")
    word2int = pickle.load(word2int)
    weights = np.ones(len(word2int))
    weights[word2int['<pad>']] = 0.0
    # Checkpoint
    checkpoint_path = os.getcwd()+"/results/trainingWeights/wphoenix-fusion-RGBFLOW-LSTM-60s-60_"+args.rUnit+"_"+args.name+".ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
                        checkpoint_path, verbose=1, save_weights_only=True)

    if args.solver == 'adam':
        loss = weighted_categorical_crossentropy(weights)
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate, cp_callback]
        adam = optimizers.Adam(lr=args.lr, 
                               beta_1=0.9, 
                               beta_2=0.999, 
                               epsilon=1e-08, 
                               decay=0.0, 
                               clipnorm=1., 
                               clipvalue=5)
        model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
        model.fit_generator(generator=training_generator,
                                      use_multiprocessing=True,
                                      workers=4,
                                      epochs=args.epochs,
                                      callbacks=callbacks_list)#validation_data=validation_generator,
    
    elif args.solver == 'sgd':
        loss = weighted_categorical_crossentropy(weights)
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate,cp_callback]
        sgd = optimizers.SGD(lr=args.lr, 
                             decay=0, 
                             momentum=args.momentum, 
                             nesterov=True, 
                             clipnorm=1., 
                             clipvalue=0.5)
        model.compile(optimizer=sgd, loss=loss, metrics=['accuracy'])
        model.fit_generator(generator=training_generator,
                                      use_multiprocessing=True,
                                      workers=4,
                                      epochs=args.epochs,
                                      callbacks=callbacks_list)#validation_data=validation_generator,
        
    else:
        pass
    
    return model


