from models.languageModels.translationModels import *
from models.learningModels import learningKerasModels
from utils import generators
import tensorflow as tf

def main(args, partition):
    if args.rUnit == 'LSTM':
        if args.name=='Top':
            model, encoder_model, decoder_model = signs2textLSTMTop(inputShape=args.inputShape,  
                                                               wDecay=args.wDecay,
                                                               Tx=args.Tx, 
                                                               Ty=args.Ty, 
                                                               denseUnits = args.denseUnits,  
                                                               encoder_unitLow_size=args.encoder_unitLow_size,
                                                               encoder_unitHigh_size=args.encoder_unitHigh_size,     
                                                               embedding_size=args.embedding_size, 
                                                               vocab_out_size=args.vocab_out_size,
                                                               dropout=args.dropout,
                                                               recurrent_dropout=args.recurrent_dropout,
                                                               FeedConnection=args.FeedConnection,
                                                               nFilters=args.nFilters)
        elif args.name=='Bottom':
            model, encoder_model, decoder_model = signs2textLSTMBottom(inputShape=args.inputShape,  
                                                               wDecay=args.wDecay,
                                                               Tx=args.Tx, 
                                                               Ty=args.Ty, 
                                                               denseUnits = args.denseUnits,  
                                                               encoder_unitLow_size=args.encoder_unitLow_size,
                                                               encoder_unitHigh_size=args.encoder_unitHigh_size,     
                                                               embedding_size=args.embedding_size, 
                                                               vocab_out_size=args.vocab_out_size,
                                                               dropout=args.dropout,
                                                               recurrent_dropout=args.recurrent_dropout,
                                                               FeedConnection=args.FeedConnection,
                                                               nFilters=args.nFilters)
        elif args.name=='Double':
            model, encoder_model, decoder_model = signs2textLSTMDouble(inputShape=args.inputShape,  
                                                               wDecay=args.wDecay,
                                                               Tx=args.Tx, 
                                                               Ty=args.Ty, 
                                                               denseUnits = args.denseUnits,  
                                                               encoder_unitLow_size=args.encoder_unitLow_size,
                                                               encoder_unitHigh_size=args.encoder_unitHigh_size,     
                                                               embedding_size=args.embedding_size, 
                                                               vocab_out_size=args.vocab_out_size,
                                                               dropout=args.dropout,
                                                               recurrent_dropout=args.recurrent_dropout,
                                                               FeedConnection=args.FeedConnection,
                                                               nFilters=args.nFilters)
    elif args.rUnit == 'GRU':
        if args.name=='Top':
            model, encoder_model, decoder_model = signs2textGRUTop(inputShape=args.inputShape,  
                                                               wDecay=args.wDecay,
                                                               Tx=args.Tx, 
                                                               Ty=args.Ty, 
                                                               denseUnits = args.denseUnits,  
                                                               encoder_unitLow_size=args.encoder_unitLow_size,
                                                               encoder_unitHigh_size=args.encoder_unitHigh_size,     
                                                               embedding_size=args.embedding_size, 
                                                               vocab_out_size=args.vocab_out_size,
                                                               dropout=args.dropout,
                                                               recurrent_dropout=args.recurrent_dropout,
                                                               FeedConnection=args.FeedConnection,
                                                               nFilters=args.nFilters)
        elif args.name=='Bottom':
            model, encoder_model, decoder_model = signs2textGRUBottom(inputShape=args.inputShape,  
                                                               wDecay=args.wDecay,
                                                               Tx=args.Tx, 
                                                               Ty=args.Ty, 
                                                               denseUnits = args.denseUnits,  
                                                               encoder_unitLow_size=args.encoder_unitLow_size,
                                                               encoder_unitHigh_size=args.encoder_unitHigh_size,     
                                                               embedding_size=args.embedding_size, 
                                                               vocab_out_size=args.vocab_out_size,
                                                               dropout=args.dropout,
                                                               recurrent_dropout=args.recurrent_dropout,
                                                               FeedConnection=args.FeedConnection,
                                                               nFilters=args.nFilters)
        elif args.name=='Double':
            model, encoder_model, decoder_model = signs2textGRUDouble(inputShape=args.inputShape,  
                                                               wDecay=args.wDecay,
                                                               Tx=args.Tx, 
                                                               Ty=args.Ty, 
                                                               denseUnits = args.denseUnits,  
                                                               encoder_unitLow_size=args.encoder_unitLow_size,
                                                               encoder_unitHigh_size=args.encoder_unitHigh_size,     
                                                               embedding_size=args.embedding_size, 
                                                               vocab_out_size=args.vocab_out_size,
                                                               dropout=args.dropout,
                                                               recurrent_dropout=args.recurrent_dropout,
                                                               FeedConnection=args.FeedConnection,
                                                               nFilters=args.nFilters)
    
    print(model.summary())
    print(encoder_model.summary())
    print(decoder_model.summary())
    ##################################################################################
    # Load weights
    #latest = tf.train.latest_checkpoint(checkpoint_dir)
    #checkpoint_path = "/home/jota/project2/results/trainingWeights/phoenix-PRUEBA-RGB-LSTM-128s-60_LSTM_Top.ckpt"
    #model.load_weights(checkpoint_path)
    # Generators 
    print('Launching generators ...')  
    training_generator = generators.DataGeneratorNMSLTp(partition['train'], 
                                                       path=args.dataNLP, 
                                                       batch_size=args.batchSize,
                                                       type_='train', 
                                                       dim=[args.inputShape[0],
                                                            args.inputShape[1],
                                                            args.inputShape[2],
                                                            args.inputShape[3],
                                                            args.Ty,
                                                            args.vocab_out_size])
    validation_generator = generators.DataGeneratorNMSLTp(partition['dev'],
                                                         path=args.dataNLP, 
                                                         batch_size=args.batchSize,
                                                         type_='dev', 
                                                         dim=[args.inputShape[0],
                                                              args.inputShape[1],
                                                              args.inputShape[2],
                                                              args.inputShape[3],
                                                              args.Ty,
                                                              args.vocab_out_size]) 
    ##################################################################################
    # Training Model
    #-----------------------------------Train----------------------------------------#    
    print('Training ...')
    model = learningKerasModels.learningSigns(model, training_generator, validation_generator, args) 
    
    ##################################################################################
    return model, encoder_model, decoder_model