import keras
import numpy as np
import tensorflow as tf


class DataGeneratorNMSLTp(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, type_, batch_size=1, dim=[60,210,260,3,53,3003],
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.path = path
        self.type_ = type_
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim[:4]))
        X2 = np.empty((self.batch_size, self.dim[4]))
        y = np.empty((self.batch_size, *self.dim[4:]), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            name = ID.split('/')[-1].split('.')[-2]
            # Store sample
            X1[i,] = np.load(ID)
            X2[i,] = np.loadtxt(self.path+"source/"+self.type_+'/'+name)
            # Store target
            y[i,] = np.loadtxt(self.path+"target/"+self.type_+'/'+name)

        return [X1, X2], y      

class DataGeneratorNMSLT(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path, batch_size=4, dim=[128,224,224,3,10,67],
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim[:4]))
        X2 = np.empty((self.batch_size, self.dim[4]))
        y = np.empty((self.batch_size, *self.dim[4:]), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            class_ = int(ID.split('/')[-2])
            # Store sample
            X1[i,] = np.load(ID)
            X2[i,] = np.loadtxt(self.path+"source/"+str(class_))
            # Store target
            y[i,] = np.loadtxt(self.path+"target/"+str(class_))

        return [X1, X2], y    

    
