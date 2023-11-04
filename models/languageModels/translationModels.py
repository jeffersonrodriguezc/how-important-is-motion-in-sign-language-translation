from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, LSTM, Dot, Dropout, Concatenate, Multiply, Conv3D, MaxPooling3D
from tensorflow.keras.layers import Input, TimeDistributed, Embedding, RepeatVector, Lambda, Bidirectional
from tensorflow.keras.layers import Flatten, Reshape, Permute, Activation, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

#from keras.utils.vis_utils import plot_model
import numpy as np

K.set_image_data_format("channels_last")

def VGG16wlayer1(shape, name=None, dtype=tf.float32, partition_info=None):
    value = np.load('/home/jota/project2/models/weights/weightsLayer1.npy')
    return K.variable(value, name=name)

def VGG16wlayer2(shape, name=None, dtype=tf.float32, partition_info=None):
    value = np.load('/home/jota/project2/models/weights/weightsLayer2.npy')
    return K.variable(value, name=name, dtype=dtype)

def VGG16wlayer3(shape, name=None, dtype=tf.float32, partition_info=None):
    value = np.load('/home/jota/project2/models/weights/weightsLayer3.npy')
    return K.variable(value, name=name, dtype=dtype)

def VGG16wlayer4(shape, name=None, dtype=tf.float32, partition_info=None):
    value = np.load('/home/jota/project2/models/weights/weightsLayer4.npy')
    return K.variable(value, name=name, dtype=dtype)

def VGG16wlayer5(shape, name=None, dtype=tf.float32, partition_info=None):
    value = np.load('/home/jota/project2/models/weights/weightsLayer5.npy')
    return K.variable(value, name=name, dtype=dtype)

def LTCSign(inputs, wDecay, nFilters):
    """ 
    LTC
    """
    # Block 1
    X = Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1), kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv1')(inputs)
    X = BatchNormalization(axis=4, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X)
    # Block 2
    X = Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1), kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv2')(X)
    X = BatchNormalization(axis=4, name='bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X)
    # Block 3
    X = Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1), kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv3')(X)
    X = BatchNormalization(axis=4, name='bn_conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X)
    # Block 4
    X = Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1), kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv4')(X)
    X = BatchNormalization(axis=4, name='bn_conv4')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X) #1,1,1 for 112x112 images
    # Block 5
    X = Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv5')(X)
    X = BatchNormalization(axis=4, name='bn_conv5')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X) #1,1,2 for 112x112 images
    # Block 6
    X = Conv3D(filters=nFilters, kernel_size=(3,3,3), strides=(1,1,1), kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv6')(X)
    X = BatchNormalization(axis=4, name='bn_conv6')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X) #1,3,2 for 112x112 images
    # Final processing 
    # Reshape to (Batch, Dim(Tx_i), Tx)
    X = Reshape((X.shape[1]*X.shape[2]*X.shape[3],X.shape[4]))(X)
    # Permute to (Batch, Tx, Dim(Tx_i))
    X = Permute((2,1))(X)
    
    return X


def WLTCSign(inputs, wDecay, nFilters):
    """ 
    LTC
    """
    # Block 1
    X = Conv3D(filters=32,kernel_size=(3,3,3),strides=(1,1,1),kernel_initializer=VGG16wlayer1, kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv1')(inputs)
    X = BatchNormalization(axis=4, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X)
    # Block 2
    X = Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1),kernel_initializer=VGG16wlayer2, kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv2')(X)
    X = BatchNormalization(axis=4, name='bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X)
    # Block 3
    X = Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1),kernel_initializer=VGG16wlayer3, kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv3')(X)
    X = BatchNormalization(axis=4, name='bn_conv3')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X)
    # Block 4
    X = Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1),kernel_initializer=VGG16wlayer4, kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv4')(X)
    X = BatchNormalization(axis=4, name='bn_conv4')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X) #1,1,1 for 112x112 images
    # Block 5
    X = Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1),kernel_initializer=VGG16wlayer5, kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv5')(X)
    X = BatchNormalization(axis=4, name='bn_conv5')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X) #1,1,2 for 112x112 images
    # Block 6
    X = Conv3D(filters=nFilters, kernel_size=(3,3,3), strides=(1,1,1), kernel_regularizer=l2(wDecay),data_format='channels_last', name='conv6')(X)
    X = BatchNormalization(axis=4, name='bn_conv6')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D(pool_size=(2,2,2), strides=(1,2,2), data_format='channels_last')(X) #1,3,2 for 112x112 images
    # Final processing 
    # Reshape to (Batch, Dim(Tx_i), Tx)
    X = Reshape((X.shape[1]*X.shape[2]*X.shape[3],X.shape[4]))(X)
    # Permute to (Batch, Tx, Dim(Tx_i))
    X = Permute((2,1))(X)
    
    return X

def signs2textLSTMDouble(inputShape=(128,224,224,3),  
            wDecay=0.0005,
            Tx=128, 
            Ty=10, 
            denseUnits = 512,  
            encoder_input_size=2016,
            decoder_input_size=10,
            encoder_unitLow_size=128,
            encoder_unitHigh_size=278,     
            embedding_size=300, 
            vocab_out_size=67,
            dropout=0.3,
            recurrent_dropout=0.3,
            FeedConnection=True,
            nFilters=128):
    ############################################################################
    # ----------------------- Shared Layers -----------------------------------#
    intermediate_att = LuongAttention(encoder_unitLow_size*2)
    high_att = LuongAttention(encoder_unitHigh_size*2)
    # Instantiate the Embedding layer.
    x_embed = Embedding(vocab_out_size, embedding_size, mask_zero=True)
    # Instantiate the LSTM layers.
    encoder_low_layer = Bidirectional(LSTM(encoder_unitLow_size, 
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       stateful=False), 
                                       merge_mode="concat",
                                       name='BiLSTM1Enc')

    encoder_high_layer = Bidirectional(LSTM(encoder_unitHigh_size, 
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        stateful=False), 
                                        merge_mode="concat",
                                        name='BiLSTM2Enc')

    decoder_low_layer = LSTM(encoder_unitLow_size*2, 
                         return_state = True, 
                         return_sequences = True,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         stateful=False,
                         name='BiLSTM1Dec')
    
    decoder_high_layer = LSTM(encoder_unitHigh_size*2, 
                          return_state = True, 
                          return_sequences = True,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          stateful=False,
                          name='BiLSTM2Dec')
    # Instantiate Dense layers.
    dense1 = Dense(denseUnits, name='dense1')
    dense2 = Dense(denseUnits*2, name='dense2')
    dense_embbed = Dense(embedding_size, name='denseEmbedd')
    dense = Dense(vocab_out_size, activation='softmax', name='dense_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    ############################################################################
    #-------------------- Feature representation (LTC) ------------------------#
    # Dim --> (Batch, Tx, W, H, C)
    encoder_input = Input(inputShape)
    features = WLTCSign(encoder_input, wDecay, nFilters)
    # Coding volume with dense layers
    coded_features = dense1(features) 
    coded_features = Dropout(dropout)(coded_features)
    coded_features = dense2(coded_features)
    coded_features = Dropout(dropout)(coded_features)
    # Coding volume to embedding size (Batch, Tx, EmbeddSize)
    coded_features = dense_embbed(coded_features) 
    #--------------------------- Encoder Model --------------------------------#
    # Define Deep Bi-LSTM Enconder
    # Recurrent low coding - (Batch, Tx, encoder_unitLow_size*2)
    outputs_encoder_low, low_h_forward, low_c_forward, low_h_backward, low_c_backward = encoder_low_layer(coded_features)
    # Recurrent high coding - (Batch, Tx, encoder_unitLow_size*2+embedding_size)
    outputs_encoder_high, high_h_forward, high_c_forward, high_h_backward, high_c_backward  = encoder_high_layer(outputs_encoder_low)
    # Concatenate the hidden states in each layer.
    low_hidden = Concatenate(axis=-1)([low_h_forward,low_h_backward])
    low_cell = Concatenate(axis=-1)([low_c_forward,low_c_backward])
    low_states = [low_hidden,low_cell]
    
    high_hidden = Concatenate(axis=-1)([high_h_forward,high_h_backward])
    high_cell = Concatenate(axis=-1)([high_c_forward,high_c_backward])
    high_states = [high_hidden,high_cell]
    
    
    encoder_model = Model(inputs=encoder_input, outputs=[outputs_encoder_low, 
                                                  low_hidden,
                                                  low_cell,       
                                                  outputs_encoder_high,
                                                  high_hidden,       
                                                  high_cell], name='Encoder_model')
    #--------------------------- Decoder Model --------------------------------#
    decoder_input = Input(shape=(Ty,), name='decoder_inputs')
    decoder_low_initial_states = low_states
    decoder_high_initial_states = high_states
    # Compute the embedding representation
    # Dim --> (Batch, 10, 300)
    input_embed = x_embed(decoder_input)
    low_outputs_decoder, _, _ = decoder_low_layer(input_embed, initial_state=decoder_low_initial_states)
    context_low, alignment_low = intermediate_att(low_outputs_decoder, outputs_encoder_low)
    decoder_concat_intermediate = Concatenate(axis=-1, name='intermediate_concat_layer')([low_outputs_decoder, context_low])
    high_outputs_decoder, _, _ = decoder_high_layer(decoder_concat_intermediate, initial_state=decoder_high_initial_states)
    context_high, alignment_high = high_att(high_outputs_decoder, outputs_encoder_high)
    decoder_concat_high = Concatenate(axis=-1, name='high_concat_layer')([high_outputs_decoder, context_high])
    # Outputs. Dim --> (Batch, Ty, Dim(vocab))
    decoder_outputs = dense_time(decoder_concat_high)
    
    model = Model(inputs = [encoder_input, decoder_input], outputs = [decoder_outputs], name='model')
    #---------------------------- Decoder Inference Model ---------------------#
    batch_size = 1
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1), name='decoder_index_word_inputs')
    encoder_inf_low_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitLow_size*2), name='encoder_inf_low_outputs')
    decoder_low_init_state_h = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_h')
    decoder_low_init_state_c = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_c')
    encoder_inf_high_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitHigh_size*2), name='encoder_inf_high_outputs')
    decoder_init_high_state_h = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_h')
    decoder_init_high_state_c = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_c')
    # Apply embed to decoder inference outputs
    # Dim --> (1,300)
    input_embed_inf = x_embed(decoder_inf_inputs)
    decoder_inf_low_out, decoder_inf_low_state_h, decoder_inf_low_state_c = decoder_low_layer(input_embed_inf, 
                                                                                              initial_state=[decoder_low_init_state_h,
                                                                                                             decoder_low_init_state_c])
    # Intermediate attention
    context_low_inf, alignment_low_inf = intermediate_att(decoder_inf_low_out, encoder_inf_low_outputs)
    
    decoder_inf_low_concat = Concatenate(axis=-1, name='concat_inf_intermediate')([decoder_inf_low_out, context_low_inf])
    decoder_inf_high_out, decoder_inf_high_state_h, decoder_inf_high_state_c = decoder_high_layer(decoder_inf_low_concat, 
                                                                                                  initial_state=[decoder_init_high_state_h,decoder_init_high_state_c])
    # High attention
    context_high_inf, alignment_high_inf = high_att(decoder_inf_high_out, encoder_inf_high_outputs)
    decoder_inf_high_concat = Concatenate(axis=-1, name='concat_inf_high')([decoder_inf_high_out, context_high_inf])
    decoder_outputs_inf = TimeDistributed(dense)(decoder_inf_high_concat)
    decoder_model = Model(inputs=[encoder_inf_low_outputs, 
                                  decoder_low_init_state_h,
                                  decoder_low_init_state_c, 
                                  encoder_inf_high_outputs, 
                                  decoder_init_high_state_h,
                                  decoder_init_high_state_c, 
                                  decoder_inf_inputs],
                          outputs=[decoder_outputs_inf, 
                                   decoder_inf_low_state_h,
                                   decoder_inf_low_state_c,
                                   alignment_low_inf,
                                   decoder_inf_high_state_h,
                                   decoder_inf_high_state_c,
                                   alignment_high_inf],
                         name="decoder_model")
    
    # Generate the network architecture.
    #plot_model(model, to_file='model_attention.png', show_shapes=True)
    #plot_model(encoder_model, to_file='encoder_model_attention.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model_attention.png', show_shapes=True) 
    
    return model, encoder_model, decoder_model 


def signs2textGRUDouble(inputShape=(128,224,224,3),  
            wDecay=0.0005,
            Tx=128, 
            Ty=10, 
            denseUnits = 512,  
            encoder_input_size=2016,
            decoder_input_size=10,
            encoder_unitLow_size=128,
            encoder_unitHigh_size=278,     
            embedding_size=300, 
            vocab_out_size=67,
            dropout=0.3,
            recurrent_dropout=0.3,
            FeedConnection=True,
            nFilters=128):
    ############################################################################
    # ----------------------- Shared Layers -----------------------------------#
    intermediate_att = LuongAttention(encoder_unitLow_size*2)
    high_att = LuongAttention(encoder_unitHigh_size*2)
    # Instantiate the Embedding layer.
    x_embed = Embedding(vocab_out_size, embedding_size, mask_zero=True)
    # Instantiate the GRU layers.
    encoder_low_layer = Bidirectional(GRU(encoder_unitLow_size, 
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       stateful=False), 
                                       merge_mode="concat",
                                       name='BiGRU1Enc')

    encoder_high_layer = Bidirectional(GRU(encoder_unitHigh_size, 
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        stateful=False), 
                                        merge_mode="concat",
                                        name='BiGRU2Enc')

    decoder_low_layer = GRU(encoder_unitLow_size*2, 
                         return_state = True, 
                         return_sequences = True,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         stateful=False,
                         name='BiGRU1Dec')
    
    decoder_high_layer = GRU(encoder_unitHigh_size*2, 
                          return_state = True, 
                          return_sequences = True,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          stateful=False,
                          name='BiGRU2Dec')
    # Instantiate Dense layers.
    dense1 = Dense(denseUnits, name='dense1')
    dense2 = Dense(denseUnits*2, name='dense2')
    dense_embbed = Dense(embedding_size, name='denseEmbedd')
    dense = Dense(vocab_out_size, activation='softmax', name='dense_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    ############################################################################
    #-------------------- Feature representation (LTC) ------------------------#
    # Dim --> (Batch, Tx, W, H, C)
    encoder_input = Input(inputShape)
    features = LTCSign(encoder_input, wDecay)
    # Coding volume with dense layers
    coded_features = dense1(features) 
    coded_features = Dropout(dropout)(coded_features)
    coded_features = dense2(coded_features)
    coded_features = Dropout(dropout)(coded_features)
    # Coding volume to embedding size (Batch, Tx, EmbeddSize)
    coded_features = dense_embbed(coded_features) 
    #--------------------------- Encoder Model --------------------------------#
    # Define Deep Bi-GRU Enconder
    # Recurrent low coding - (Batch, Tx, encoder_unitLow_size*2)
    outputs_encoder_low, low_h_forward, low_h_backward = encoder_low_layer(coded_features)
    # Recurrent high coding - (Batch, Tx, encoder_unitLow_size*2+embedding_size)
    outputs_encoder_high, high_h_forward, high_h_backward = encoder_high_layer(outputs_encoder_low)
    # Concatenate the hidden states in each layer.
    low_hidden = Concatenate(axis=-1)([low_h_forward,low_h_backward])
    high_hidden = Concatenate(axis=-1)([high_h_forward,high_h_backward])
    
    encoder_model = Model(inputs=encoder_input, outputs=[outputs_encoder_low, 
                                                  low_hidden,
                                                  outputs_encoder_high,
                                                  high_hidden], name='Encoder_model')
    #--------------------------- Decoder Model --------------------------------#
    decoder_input = Input(shape=(Ty,), name='decoder_inputs')
    decoder_low_initial_states = low_hidden
    decoder_high_initial_states = high_hidden
    # Compute the embedding representation
    # Dim --> (Batch, 10, 300)
    input_embed = x_embed(decoder_input)
    low_outputs_decoder, _ = decoder_low_layer(input_embed, initial_state=decoder_low_initial_states)
    context_low, alignment_low = intermediate_att(low_outputs_decoder, outputs_encoder_low)
    
    if FeedConnection:
        decoder_concat_intermediate = Concatenate(axis=-1, name='intermediate_concat_layer')([low_outputs_decoder, context_low])
        high_outputs_decoder, _ = decoder_high_layer(decoder_concat_intermediate, initial_state=decoder_high_initial_states)
    else:
        high_outputs_decoder, _ = decoder_high_layer(context_low, initial_state=decoder_high_initial_states)
    
    context_high, alignment_high = high_att(high_outputs_decoder, outputs_encoder_high)
    
    if FeedConnection:
        decoder_concat_high = Concatenate(axis=-1, name='high_concat_layer')([high_outputs_decoder, context_high])
        # Outputs. Dim --> (Batch, Ty, Dim(vocab))
        decoder_outputs = dense_time(decoder_concat_high)
    else:    
        decoder_outputs = dense_time(context_high)
        
    model = Model(inputs = [encoder_input, decoder_input], outputs = [decoder_outputs], name='model')
    #---------------------------- Decoder Inference Model ---------------------#
    batch_size = 1
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1), name='decoder_index_word_inputs')
    encoder_inf_low_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitLow_size*2), name='encoder_inf_low_outputs')
    decoder_low_init_state_h = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_h')
    encoder_inf_high_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitHigh_size*2), name='encoder_inf_high_outputs')
    decoder_init_high_state_h = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_h')
    # Apply embed to decoder inference outputs
    # Dim --> (1,300)
    input_embed_inf = x_embed(decoder_inf_inputs)
    decoder_inf_low_out, decoder_inf_low_state_h = decoder_low_layer(input_embed_inf, initial_state=decoder_low_init_state_h)
    # Intermediate attention
    context_low_inf, alignment_low_inf = intermediate_att(decoder_inf_low_out, encoder_inf_low_outputs)
    
    if FeedConnection:
        decoder_inf_low_concat = Concatenate(axis=-1, name='concat_inf_intermediate')([decoder_inf_low_out, context_low_inf])
        decoder_inf_high_out, decoder_inf_high_state_h = decoder_high_layer(decoder_inf_low_concat, initial_state=decoder_init_high_state_h)
    else:
        decoder_inf_high_out, decoder_inf_high_state_h = decoder_high_layer(context_low_inf, initial_state=decoder_init_high_state_h)
        
    # High attention
    context_high_inf, alignment_high_inf = high_att(decoder_inf_high_out, encoder_inf_high_outputs)
    
    if FeedConnection:
        decoder_inf_high_concat = Concatenate(axis=-1, name='concat_inf_high')([decoder_inf_high_out, context_high_inf])
        decoder_outputs_inf = TimeDistributed(dense)(decoder_inf_high_concat)
    else:
        decoder_outputs_inf = TimeDistributed(dense)(context_high_inf)        

    decoder_model = Model(inputs=[encoder_inf_low_outputs, 
                                  decoder_low_init_state_h,
                                  encoder_inf_high_outputs, 
                                  decoder_init_high_state_h,
                                  decoder_inf_inputs],
                          outputs=[decoder_outputs_inf, 
                                   decoder_inf_low_state_h,
                                   alignment_low_inf,
                                   decoder_inf_high_state_h,
                                   alignment_high_inf],
                         name="decoder_model")
    
    # Generate the network architecture.
    #plot_model(model, to_file='model_attention.png', show_shapes=True)
    #plot_model(encoder_model, to_file='encoder_model_attention.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model_attention.png', show_shapes=True) 
    
    return model, encoder_model, decoder_model 


def signs2textLSTMTop(inputShape=(128,224,224,3),  
            wDecay=0.0005,
            Tx=128, 
            Ty=10, 
            denseUnits = 512,  
            encoder_input_size=2016,
            decoder_input_size=10,
            encoder_unitLow_size=128,
            encoder_unitHigh_size=278,     
            embedding_size=300, 
            vocab_out_size=67,
            dropout=0.3,
            recurrent_dropout=0.3,
            FeedConnection=True,
            nFilters=128):
    ############################################################################
    # ----------------------- Shared Layers -----------------------------------#
    high_att = LuongAttention(encoder_unitHigh_size*2)
    # Instantiate the Embedding layer.
    x_embed = Embedding(vocab_out_size, embedding_size, mask_zero=True)
    # Instantiate the LSTM layers.
    encoder_low_layer = Bidirectional(LSTM(encoder_unitLow_size, 
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       stateful=False), 
                                       merge_mode="concat",
                                       name='BiLSTM1Enc')

    encoder_high_layer = Bidirectional(LSTM(encoder_unitHigh_size, 
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        stateful=False), 
                                        merge_mode="concat",
                                        name='BiLSTM2Enc')

    decoder_low_layer = LSTM(encoder_unitLow_size*2, 
                         return_state = True, 
                         return_sequences = True,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         stateful=False,
                         name='BiLSTM1Dec')
    
    decoder_high_layer = LSTM(encoder_unitHigh_size*2, 
                          return_state = True, 
                          return_sequences = True,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          stateful=False,
                          name='BiLSTM2Dec')
    # Instantiate Dense layers.
    dense1 = Dense(denseUnits, name='dense1')
    dense2 = Dense(denseUnits*2, name='dense2')
    dense_embbed = Dense(embedding_size, name='denseEmbedd')
    dense = Dense(vocab_out_size, activation='softmax', name='dense_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    ############################################################################
    #-------------------- Feature representation (LTC) ------------------------#
    # Dim --> (Batch, Tx, W, H, C)
    encoder_input = Input(inputShape)
    features = WLTCSign(encoder_input, wDecay, nFilters)
    # Coding volume with dense layers
    coded_features = dense1(features) 
    coded_features = Dropout(dropout)(coded_features)
    coded_features = dense2(coded_features)
    coded_features = Dropout(dropout)(coded_features)
    # Coding volume to embedding size (Batch, Tx, EmbeddSize)
    coded_features = dense_embbed(coded_features) 
    #--------------------------- Encoder Model --------------------------------#
    # Define Deep Bi-LSTM Encoder
    # Recurrent low coding - (Batch, Tx, encoder_unitLow_size*2)
    outputs_encoder_low, low_h_forward, low_c_forward, low_h_backward, low_c_backward = encoder_low_layer(coded_features)
    # Recurrent high coding - (Batch, Tx, encoder_unitLow_size*2+embedding_size)
    outputs_encoder_high, high_h_forward, high_c_forward, high_h_backward, high_c_backward  = encoder_high_layer(outputs_encoder_low)
    # Concatenate the hidden states in each layer.
    low_hidden = Concatenate(axis=-1)([low_h_forward,low_h_backward])
    low_cell = Concatenate(axis=-1)([low_c_forward,low_c_backward])
    low_states = [low_hidden,low_cell]
    
    high_hidden = Concatenate(axis=-1)([high_h_forward,high_h_backward])
    high_cell = Concatenate(axis=-1)([high_c_forward,high_c_backward])
    high_states = [high_hidden,high_cell]
    
    
    encoder_model = Model(inputs=encoder_input, outputs=[outputs_encoder_low, 
                                                  low_hidden,
                                                  low_cell,       
                                                  outputs_encoder_high,
                                                  high_hidden,       
                                                  high_cell], name='Encoder_model')
    #--------------------------- Decoder Model --------------------------------#
    decoder_input = Input(shape=(Ty,), name='decoder_inputs')
    decoder_low_initial_states = low_states
    decoder_high_initial_states = high_states
    # Compute the embedding representation
    # Dim --> (Batch, 10, 300)
    input_embed = x_embed(decoder_input)
    low_outputs_decoder, _, _ = decoder_low_layer(input_embed, initial_state=decoder_low_initial_states)
    high_outputs_decoder, _, _ = decoder_high_layer(low_outputs_decoder, initial_state=decoder_high_initial_states)
    context_high, alignment_high = high_att(high_outputs_decoder, outputs_encoder_high)
    
    if FeedConnection:
        decoder_concat_high = Concatenate(axis=-1, name='high_concat_layer')([high_outputs_decoder, context_high])
        # Outputs. Dim --> (Batch, Ty, Dim(vocab))
        decoder_outputs = dense_time(decoder_concat_high)
    else:
        decoder_outputs = dense_time(context_high)
    
    model = Model(inputs = [encoder_input, decoder_input], outputs = [decoder_outputs], name='model')
    #---------------------------- Decoder Inference Model ---------------------#
    batch_size = 1
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1), name='decoder_index_word_inputs')
    encoder_inf_low_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitLow_size*2), name='encoder_inf_low_outputs')
    decoder_low_init_state_h = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_h')
    decoder_low_init_state_c = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_c')
    encoder_inf_high_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitHigh_size*2), name='encoder_inf_high_outputs')
    decoder_init_high_state_h = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_h')
    decoder_init_high_state_c = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_c')
    # Apply embed to decoder inference outputs
    # Dim --> (1,300)
    input_embed_inf = x_embed(decoder_inf_inputs)
    decoder_inf_low_out, decoder_inf_low_state_h, decoder_inf_low_state_c = decoder_low_layer(input_embed_inf, 
                                                                                              initial_state=[decoder_low_init_state_h, decoder_low_init_state_c])    
    decoder_inf_high_out, decoder_inf_high_state_h, decoder_inf_high_state_c = decoder_high_layer(decoder_inf_low_out, 
                                                                                                  initial_state=[decoder_init_high_state_h,decoder_init_high_state_c])
    # High attention
    context_high_inf, alignment_high_inf = high_att(decoder_inf_high_out, encoder_inf_high_outputs)
    
    if FeedConnection:
        decoder_inf_high_concat = Concatenate(axis=-1, name='concat_inf_high')([decoder_inf_high_out, context_high_inf])
        decoder_outputs_inf = TimeDistributed(dense)(decoder_inf_high_concat)
    else:
        decoder_outputs_inf = TimeDistributed(dense)(context_high_inf)
        
    decoder_model = Model(inputs=[encoder_inf_low_outputs, 
                                  decoder_low_init_state_h,
                                  decoder_low_init_state_c, 
                                  encoder_inf_high_outputs, 
                                  decoder_init_high_state_h,
                                  decoder_init_high_state_c, 
                                  decoder_inf_inputs],
                          outputs=[decoder_outputs_inf, 
                                   decoder_inf_low_state_h,
                                   decoder_inf_low_state_c,
                                   decoder_inf_high_state_h,
                                   decoder_inf_high_state_c,
                                   alignment_high_inf],
                         name="decoder_model")
    
    # Generate the network architecture.
    #plot_model(model, to_file='model_attention.png', show_shapes=True)
    #plot_model(encoder_model, to_file='encoder_model_attention.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model_attention.png', show_shapes=True) 
    
    return model, encoder_model, decoder_model 

def signs2textGRUTop(inputShape=(128,224,224,3),  
            wDecay=0.0005,
            Tx=128, 
            Ty=10, 
            denseUnits = 512,  
            encoder_input_size=2016,
            decoder_input_size=10,
            encoder_unitLow_size=128,
            encoder_unitHigh_size=278,     
            embedding_size=300, 
            vocab_out_size=67,
            dropout=0.3,
            recurrent_dropout=0.3,
            FeedConnection=True,
            nFilters=128):
    ############################################################################
    # ----------------------- Shared Layers -----------------------------------#
    high_att = LuongAttention(encoder_unitHigh_size*2)
    # Instantiate the Embedding layer.
    x_embed = Embedding(vocab_out_size, embedding_size, mask_zero=True)
    # Instantiate the GRU layers.
    encoder_low_layer = Bidirectional(GRU(encoder_unitLow_size, 
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       stateful=False), 
                                       merge_mode="concat",
                                       name='BiGRU1Enc')

    encoder_high_layer = Bidirectional(GRU(encoder_unitHigh_size, 
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        stateful=False), 
                                        merge_mode="concat",
                                        name='BiGRU2Enc')

    decoder_low_layer = GRU(encoder_unitLow_size*2, 
                         return_state = True, 
                         return_sequences = True,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         stateful=False,
                         name='BiGRU1Dec')
    
    decoder_high_layer = GRU(encoder_unitHigh_size*2, 
                          return_state = True, 
                          return_sequences = True,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          stateful=False,
                          name='BiGRU2Dec')
    # Instantiate Dense layers.
    dense1 = Dense(denseUnits, name='dense1')
    dense2 = Dense(denseUnits*2, name='dense2')
    dense_embbed = Dense(embedding_size, name='denseEmbedd')
    dense = Dense(vocab_out_size, activation='softmax', name='dense_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    ############################################################################
    #-------------------- Feature representation (LTC) ------------------------#
    # Dim --> (Batch, Tx, W, H, C)
    encoder_input = Input(inputShape)
    features = LTCSign(encoder_input, wDecay)
    # Coding volume with dense layers
    coded_features = dense1(features) 
    coded_features = Dropout(dropout)(coded_features)
    coded_features = dense2(coded_features)
    coded_features = Dropout(dropout)(coded_features)
    # Coding volume to embedding size (Batch, Tx, EmbeddSize)
    coded_features = dense_embbed(coded_features) 
    #--------------------------- Encoder Model --------------------------------#
    # Define Deep Bi-GRU Enconder
    # Recurrent low coding - (Batch, Tx, encoder_unitLow_size*2)
    outputs_encoder_low, low_h_forward, low_h_backward = encoder_low_layer(coded_features)
    # Recurrent high coding - (Batch, Tx, encoder_unitLow_size*2+embedding_size)
    outputs_encoder_high, high_h_forward, high_h_backward = encoder_high_layer(outputs_encoder_low)
    # Concatenate the hidden states in each layer.
    low_hidden = Concatenate(axis=-1)([low_h_forward,low_h_backward])
    high_hidden = Concatenate(axis=-1)([high_h_forward,high_h_backward])
    
    encoder_model = Model(inputs=encoder_input, outputs=[outputs_encoder_low, 
                                                  low_hidden,
                                                  outputs_encoder_high,
                                                  high_hidden], name='Encoder_model')
    #--------------------------- Decoder Model --------------------------------#
    decoder_input = Input(shape=(Ty,), name='decoder_inputs')
    decoder_low_initial_states = low_hidden
    decoder_high_initial_states = high_hidden
    # Compute the embedding representation
    # Dim --> (Batch, 10, 300)
    input_embed = x_embed(decoder_input)
    low_outputs_decoder, _ = decoder_low_layer(input_embed, initial_state=decoder_low_initial_states)
    high_outputs_decoder, _ = decoder_high_layer(low_outputs_decoder, initial_state=decoder_high_initial_states)
    context_high, alignment_high = high_att(high_outputs_decoder, outputs_encoder_high)
    
    if FeedConnection:
        decoder_concat_high = Concatenate(axis=-1, name='high_concat_layer')([high_outputs_decoder, context_high])
        # Outputs. Dim --> (Batch, Ty, Dim(vocab))
        decoder_outputs = dense_time(decoder_concat_high)
    else:    
        decoder_outputs = dense_time(context_high)
        
    model = Model(inputs = [encoder_input, decoder_input], outputs = [decoder_outputs], name='model')
    #---------------------------- Decoder Inference Model ---------------------#
    batch_size = 1
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1), name='decoder_index_word_inputs')
    encoder_inf_low_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitLow_size*2), name='encoder_inf_low_outputs')
    decoder_low_init_state_h = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_h')
    encoder_inf_high_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitHigh_size*2), name='encoder_inf_high_outputs')
    decoder_init_high_state_h = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_h')
    # Apply embed to decoder inference outputs
    # Dim --> (1,300)
    input_embed_inf = x_embed(decoder_inf_inputs)
    decoder_inf_low_out, decoder_inf_low_state_h = decoder_low_layer(input_embed_inf,initial_state=decoder_low_init_state_h)
    decoder_inf_high_out, decoder_inf_high_state_h = decoder_high_layer(decoder_inf_low_out,     initial_state=decoder_init_high_state_h)
    # High attention
    context_high_inf, alignment_high_inf = high_att(decoder_inf_high_out, encoder_inf_high_outputs)
    
    if FeedConnection:
        decoder_inf_high_concat = Concatenate(axis=-1, name='concat_inf_high')([decoder_inf_high_out, context_high_inf])
        decoder_outputs_inf = TimeDistributed(dense)(decoder_inf_high_concat)
    else:
        decoder_outputs_inf = TimeDistributed(dense)(context_high_inf)
        
    decoder_model = Model(inputs=[encoder_inf_low_outputs, 
                                  decoder_low_init_state_h,
                                  encoder_inf_high_outputs, 
                                  decoder_init_high_state_h,
                                  decoder_inf_inputs],
                          outputs=[decoder_outputs_inf, 
                                   decoder_inf_low_state_h,
                                   decoder_inf_high_state_h,
                                   alignment_high_inf],
                         name="decoder_model")
    
    # Generate the network architecture.
    #plot_model(model, to_file='model_attention.png', show_shapes=True)
    #plot_model(encoder_model, to_file='encoder_model_attention.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model_attention.png', show_shapes=True) 
    
    return model, encoder_model, decoder_model 


def signs2textLSTMBottom(inputShape=(128,224,224,3),  
            wDecay=0.0005,
            Tx=128, 
            Ty=10, 
            denseUnits = 512,  
            encoder_input_size=2016,
            decoder_input_size=10,
            encoder_unitLow_size=128,
            encoder_unitHigh_size=278,     
            embedding_size=300, 
            vocab_out_size=67,
            dropout=0.3,
            recurrent_dropout=0.3,
            FeedConnection=True,
            nFilters=128):
    ############################################################################
    # ----------------------- Shared Layers -----------------------------------#
    intermediate_att = LuongAttention(encoder_unitLow_size*2)
    # Instantiate the Embedding layer.
    x_embed = Embedding(vocab_out_size, embedding_size, mask_zero=True)
    # Instantiate the LSTM layers.
    encoder_low_layer = Bidirectional(LSTM(encoder_unitLow_size, 
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       stateful=False), 
                                       merge_mode="concat",
                                       name='BiLSTM1Enc')

    encoder_high_layer = Bidirectional(LSTM(encoder_unitHigh_size, 
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        stateful=False), 
                                        merge_mode="concat",
                                        name='BiLSTM2Enc')

    decoder_low_layer = LSTM(encoder_unitLow_size*2, 
                         return_state = True, 
                         return_sequences = True,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         stateful=False,
                         name='BiLSTM1Dec')
    
    decoder_high_layer = LSTM(encoder_unitHigh_size*2, 
                          return_state = True, 
                          return_sequences = True,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          stateful=False,
                          name='BiLSTM2Dec')
    # Instantiate Dense layers.
    dense1 = Dense(denseUnits, name='dense1')
    dense2 = Dense(denseUnits*2, name='dense2')
    dense_embbed = Dense(embedding_size, name='denseEmbedd')
    dense = Dense(vocab_out_size, activation='softmax', name='dense_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    ############################################################################
    #-------------------- Feature representation (LTC) ------------------------#
    # Dim --> (Batch, Tx, W, H, C)
    encoder_input = Input(inputShape)
    features = LTCSign(encoder_input, wDecay, nFilters)
    # Coding volume with dense layers
    coded_features = dense1(features) 
    coded_features = Dropout(dropout)(coded_features)
    coded_features = dense2(coded_features)
    coded_features = Dropout(dropout)(coded_features)
    # Coding volume to embedding size (Batch, Tx, EmbeddSize)
    coded_features = dense_embbed(coded_features) 
    #--------------------------- Encoder Model --------------------------------#
    # Define Deep Bi-LSTM Enconder
    # Recurrent low coding - (Batch, Tx, encoder_unitLow_size*2)
    outputs_encoder_low, low_h_forward, low_c_forward, low_h_backward, low_c_backward = encoder_low_layer(coded_features)
    # Recurrent high coding - (Batch, Tx, encoder_unitLow_size*2+embedding_size)
    outputs_encoder_high, high_h_forward, high_c_forward, high_h_backward, high_c_backward  = encoder_high_layer(outputs_encoder_low)
    # Concatenate the hidden states in each layer.
    low_hidden = Concatenate(axis=-1)([low_h_forward,low_h_backward])
    low_cell = Concatenate(axis=-1)([low_c_forward,low_c_backward])
    low_states = [low_hidden,low_cell]
    
    high_hidden = Concatenate(axis=-1)([high_h_forward,high_h_backward])
    high_cell = Concatenate(axis=-1)([high_c_forward,high_c_backward])
    high_states = [high_hidden,high_cell]
    
    
    encoder_model = Model(inputs=encoder_input, outputs=[outputs_encoder_low, 
                                                  low_hidden,
                                                  low_cell,       
                                                  outputs_encoder_high,
                                                  high_hidden,       
                                                  high_cell], name='Encoder_model')
    #--------------------------- Decoder Model --------------------------------#
    decoder_input = Input(shape=(Ty,), name='decoder_inputs')
    decoder_low_initial_states = low_states
    decoder_high_initial_states = high_states
    # Compute the embedding representation
    # Dim --> (Batch, 10, 300)
    input_embed = x_embed(decoder_input)
    low_outputs_decoder, _, _ = decoder_low_layer(input_embed, initial_state=decoder_low_initial_states)
    context_low, alignment_low = intermediate_att(low_outputs_decoder, outputs_encoder_low)
    
    if FeedConnection:
        decoder_concat_intermediate = Concatenate(axis=-1, name='intermediate_concat_layer')([low_outputs_decoder, context_low])
        high_outputs_decoder, _, _ = decoder_high_layer(decoder_concat_intermediate, initial_state=decoder_high_initial_states)
    else:
        high_outputs_decoder, _, _ = decoder_high_layer(context_low, initial_state=decoder_high_initial_states) 
    
    # Outputs. Dim --> (Batch, Ty, Dim(vocab))
    decoder_outputs = dense_time(high_outputs_decoder)
    
    model = Model(inputs = [encoder_input, decoder_input], outputs = [decoder_outputs], name='model')
    #---------------------------- Decoder Inference Model ---------------------#
    batch_size = 1
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1), name='decoder_index_word_inputs')
    encoder_inf_low_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitLow_size*2), name='encoder_inf_low_outputs')
    decoder_low_init_state_h = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_h')
    decoder_low_init_state_c = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_c')
    encoder_inf_high_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitHigh_size*2), name='encoder_inf_high_outputs')
    decoder_init_high_state_h = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_h')
    decoder_init_high_state_c = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_c')
    # Apply embed to decoder inference outputs
    # Dim --> (1,300)
    input_embed_inf = x_embed(decoder_inf_inputs)
    decoder_inf_low_out, decoder_inf_low_state_h, decoder_inf_low_state_c = decoder_low_layer(input_embed_inf, 
                                                                                              initial_state=[decoder_low_init_state_h,
                                                                                                             decoder_low_init_state_c])
    # Intermediate attention
    context_low_inf, alignment_low_inf = intermediate_att(decoder_inf_low_out, encoder_inf_low_outputs)
    
    if FeedConnection:
        decoder_inf_low_concat = Concatenate(axis=-1, name='concat_inf_intermediate')([decoder_inf_low_out, context_low_inf])
        decoder_inf_high_out, decoder_inf_high_state_h, decoder_inf_high_state_c = decoder_high_layer(decoder_inf_low_concat, 
                                                                                                  initial_state=[decoder_init_high_state_h,decoder_init_high_state_c])
    else:
        decoder_inf_high_out, decoder_inf_high_state_h, decoder_inf_high_state_c = decoder_high_layer(context_low_inf, 
                                                                                                  initial_state=[decoder_init_high_state_h,decoder_init_high_state_c])
    decoder_outputs_inf = TimeDistributed(dense)(decoder_inf_high_out)
    decoder_model = Model(inputs=[encoder_inf_low_outputs, 
                                  decoder_low_init_state_h,
                                  decoder_low_init_state_c, 
                                  encoder_inf_high_outputs, 
                                  decoder_init_high_state_h,
                                  decoder_init_high_state_c, 
                                  decoder_inf_inputs],
                          outputs=[decoder_outputs_inf, 
                                   decoder_inf_low_state_h,
                                   decoder_inf_low_state_c,
                                   alignment_low_inf,
                                   decoder_inf_high_state_h,
                                   decoder_inf_high_state_c],
                         name="decoder_model")
    
    # Generate the network architecture.
    #plot_model(model, to_file='model_attention.png', show_shapes=True)
    #plot_model(encoder_model, to_file='encoder_model_attention.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model_attention.png', show_shapes=True) 
    
    return model, encoder_model, decoder_model 


def signs2textGRUBottom(inputShape=(128,224,224,3),  
            wDecay=0.0005,
            Tx=128, 
            Ty=10, 
            denseUnits = 512,  
            encoder_input_size=2016,
            decoder_input_size=10,
            encoder_unitLow_size=128,
            encoder_unitHigh_size=278,     
            embedding_size=300, 
            vocab_out_size=67,
            dropout=0.3,
            recurrent_dropout=0.3,
            FeedConnection=True,
            nFilters=128):
    ############################################################################
    # ----------------------- Shared Layers -----------------------------------#
    intermediate_att = LuongAttention(encoder_unitLow_size*2)
    # Instantiate the Embedding layer.
    x_embed = Embedding(vocab_out_size, embedding_size, mask_zero=True)
    # Instantiate the GRU layers.
    encoder_low_layer = Bidirectional(GRU(encoder_unitLow_size, 
                                       return_sequences=True,
                                       return_state=True,
                                       dropout=dropout,
                                       recurrent_dropout=recurrent_dropout,
                                       stateful=False), 
                                       merge_mode="concat",
                                       name='BiGRU1Enc')

    encoder_high_layer = Bidirectional(GRU(encoder_unitHigh_size, 
                                        return_sequences=True,
                                        return_state=True,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        stateful=False), 
                                        merge_mode="concat",
                                        name='BiGRU2Enc')

    decoder_low_layer = GRU(encoder_unitLow_size*2, 
                         return_state = True, 
                         return_sequences = True,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         stateful=False,
                         name='BiGRU1Dec')
    
    decoder_high_layer = GRU(encoder_unitHigh_size*2, 
                          return_state = True, 
                          return_sequences = True,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          stateful=False,
                          name='BiGRU2Dec')
    # Instantiate Dense layers.
    dense1 = Dense(denseUnits, name='dense1')
    dense2 = Dense(denseUnits*2, name='dense2')
    dense_embbed = Dense(embedding_size, name='denseEmbedd')
    dense = Dense(vocab_out_size, activation='softmax', name='dense_layer')
    dense_time = TimeDistributed(dense, name='time_distributed_layer')
    ############################################################################
    #-------------------- Feature representation (LTC) ------------------------#
    # Dim --> (Batch, Tx, W, H, C)
    encoder_input = Input(inputShape)
    features = LTCSign(encoder_input, wDecay, nFilters)
    # Coding volume with dense layers
    coded_features = dense1(features) 
    coded_features = Dropout(dropout)(coded_features)
    coded_features = dense2(coded_features)
    coded_features = Dropout(dropout)(coded_features)
    # Coding volume to embedding size (Batch, Tx, EmbeddSize)
    coded_features = dense_embbed(coded_features) 
    #--------------------------- Encoder Model --------------------------------#
    # Define Deep Bi-LSTM Enconder
    # Recurrent low coding - (Batch, Tx, encoder_unitLow_size*2)
    outputs_encoder_low, low_h_forward, low_h_backward = encoder_low_layer(coded_features)
    # Recurrent high coding - (Batch, Tx, encoder_unitLow_size*2+embedding_size)
    outputs_encoder_high, high_h_forward, high_h_backward = encoder_high_layer(outputs_encoder_low)
    # Concatenate the hidden states in each layer.
    low_hidden = Concatenate(axis=-1)([low_h_forward,low_h_backward])
    high_hidden = Concatenate(axis=-1)([high_h_forward,high_h_backward])
    
    encoder_model = Model(inputs=encoder_input, outputs=[outputs_encoder_low, 
                                                  low_hidden,
                                                  outputs_encoder_high,
                                                  high_hidden], name='Encoder_model')
    #--------------------------- Decoder Model --------------------------------#
    decoder_input = Input(shape=(Ty,), name='decoder_inputs')
    decoder_low_initial_states = low_hidden
    decoder_high_initial_states = high_hidden
    # Compute the embedding representation
    # Dim --> (Batch, 10, 300)
    input_embed = x_embed(decoder_input)
    low_outputs_decoder, _ = decoder_low_layer(input_embed, initial_state=decoder_low_initial_states)
    context_low, alignment_low = intermediate_att(low_outputs_decoder, outputs_encoder_low)
    
    if FeedConnection:
        decoder_concat_intermediate = Concatenate(axis=-1, name='intermediate_concat_layer')([low_outputs_decoder, context_low])
        high_outputs_decoder, _ = decoder_high_layer(decoder_concat_intermediate, initial_state=decoder_high_initial_states)    
    else:
        high_outputs_decoder, _ = decoder_high_layer(context_low, initial_state=decoder_high_initial_states)
        
    # Outputs. Dim --> (Batch, Ty, Dim(vocab))
    decoder_outputs = dense_time(high_outputs_decoder)
    
    model = Model(inputs = [encoder_input, decoder_input], outputs = [decoder_outputs], name='model')
    #---------------------------- Decoder Inference Model ---------------------#
    batch_size = 1
    decoder_inf_inputs = Input(batch_shape=(batch_size, 1), name='decoder_index_word_inputs')
    encoder_inf_low_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitLow_size*2), name='encoder_inf_low_outputs')
    decoder_low_init_state_h = Input(batch_shape=(batch_size, encoder_unitLow_size*2), name='decoder_low_init_state_h')
    encoder_inf_high_outputs = Input(batch_shape=(batch_size, nFilters, encoder_unitHigh_size*2), name='encoder_inf_high_outputs')
    decoder_init_high_state_h = Input(batch_shape=(batch_size, encoder_unitHigh_size*2), name='decoder_init_high_state_h')
    # Apply embed to decoder inference outputs
    # Dim --> (1,300)
    input_embed_inf = x_embed(decoder_inf_inputs)
    decoder_inf_low_out, decoder_inf_low_state_h = decoder_low_layer(input_embed_inf, initial_state=decoder_low_init_state_h)
    # Intermediate attention
    context_low_inf, alignment_low_inf = intermediate_att(decoder_inf_low_out, encoder_inf_low_outputs)
    
    if FeedConnection:
        decoder_inf_low_concat = Concatenate(axis=-1, name='concat_inf_intermediate')([decoder_inf_low_out, context_low_inf])
        decoder_inf_high_out, decoder_inf_high_state_h = decoder_high_layer(decoder_inf_low_concat, initial_state=decoder_init_high_state_h)
    else:
        decoder_inf_high_out, decoder_inf_high_state_h = decoder_high_layer(context_low_inf, initial_state=decoder_init_high_state_h)
        
    decoder_outputs_inf = TimeDistributed(dense)(decoder_inf_high_out)
    decoder_model = Model(inputs=[encoder_inf_low_outputs, 
                                  decoder_low_init_state_h,
                                  encoder_inf_high_outputs, 
                                  decoder_init_high_state_h,
                                  decoder_inf_inputs],
                          outputs=[decoder_outputs_inf, 
                                   decoder_inf_low_state_h,
                                   alignment_low_inf,
                                   decoder_inf_high_state_h],
                         name="decoder_model")
    
    # Generate the network architecture.
    #plot_model(model, to_file='model_attention.png', show_shapes=True)
    #plot_model(encoder_model, to_file='encoder_model_attention.png', show_shapes=True)
    #plot_model(decoder_model, to_file='decoder_model_attention.png', show_shapes=True) 
    
    return model, encoder_model, decoder_model 


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(10)
 
    def call(self, decoder_output, encoder_output):
        score = tf.nn.tanh(tf.matmul(self.W1(encoder_output), self.W2(decoder_output), transpose_b=True))
        alignment = tf.nn.softmax(self.V(score), axis=2)
        alignment = tf.keras.backend.permute_dimensions(alignment, (0,2,1))
        context = tf.matmul(alignment, encoder_output)
        return context, alignment

class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.Wa = tf.keras.layers.Dense(rnn_size)
    
    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(decoder_output, self.Wa(encoder_output), transpose_b=True)
        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)
        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)
        return context, alignment
