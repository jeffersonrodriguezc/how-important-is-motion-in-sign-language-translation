import numpy as np
import pickle
import jiwer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
import nltk
import copy
import pandas as pd
nltk.download('wordnet')

def video2textDouble(model, encoder_model, decoder_model, args, partition, name='Double',search='Greedy'):
    """
    Translation for training and test data.
    """
    # Dics
    word2int = open(args.dataNLP+"dicts/word2int.pkl","rb")
    word2int = pickle.load(word2int)
    int2word = open(args.dataNLP+"dicts/int2word.pkl","rb")
    int2word = pickle.load(int2word) 
    results = defaultdict(dict)
    
    for key in partition.keys():
        print(key, ' data')
        for j, i in enumerate(partition[key]):
            print('video: ', i.split('/')[-1])
            video = np.load(i)
            video = np.expand_dims(video, axis=0)
            #print(features.shape)
            # Prediction of latent vectors for decoder step
            if args.rUnit == 'LSTM':
                enc_outputs_low, low_hidden, low_cell, enc_outputs_high, high_hidden, high_cell = encoder_model.predict(video)
                states_value_first_layer_low = [low_hidden,low_cell]
                states_value_first_layer_high = [high_hidden,high_cell]
                target_seq = np.zeros((1, 1))
                # Populate the first character of target sequence with the start character.
                target_seq[0, 0] = word2int['<bos>']
                stop_condition = False
                decoded_sentence = list()
                alignment_low_steps = list()
                alignment_high_steps = list()
                while not stop_condition:
                    output, dec_low_h, dec_low_c, alignment_low, dec_high_h, dec_high_c, alignment_high = decoder_model.predict([enc_outputs_low,states_value_first_layer_low[0],states_value_first_layer_low[1],enc_outputs_high,states_value_first_layer_high[0],states_value_first_layer_high[1],target_seq])
                    # Sample a token
                    sampled_token_index = np.argmax(output[0, -1, :])
                    sampled_char = int2word[sampled_token_index]
                    print('output,',sampled_char)
                    decoded_sentence.append(sampled_char)
                    alignment_low_steps.append(alignment_low)
                    alignment_high_steps.append(alignment_high)
                    # Exit condition: either hit max length
                    # or find stop character.
                    if (sampled_char == '<eos>' or len(decoded_sentence) > args.Ty):
                        stop_condition = True
                    if search=="Greedy":
                        target_seq = np.zeros((1, 1))
                        target_seq[0, 0] = sampled_token_index
                    states_value_first_layer_low = [dec_low_h, dec_low_c]
                    states_value_first_layer_high = [dec_high_h, dec_high_c]
                
                # Save translated sentence and aligments    
                results[key][i.split('/')[-1]] = [decoded_sentence, alignment_low_steps, alignment_high_steps]
                print(" ".join(decoded_sentence))
                decoded_sentence = list() 
                
            elif args.rUnit == 'GRU':
                pass
            
    with open(args.path2save+name+'.pkl', 'wb') as file:
            pickle.dump(results, file)        
    return results


def video2textTOP(model, encoder_model, decoder_model, args, partition, name='TOP', search='Greedy'):
    """
    Translation for training and test data.
    """
    # Dics
    word2int = open(args.dataNLP+"dicts/word2int.pkl","rb")
    word2int = pickle.load(word2int)
    int2word = open(args.dataNLP+"dicts/int2word.pkl","rb")
    int2word = pickle.load(int2word) 
    results = defaultdict(dict)
    
    for key in partition.keys():
        print(key, ' data')
        for j, i in enumerate(partition[key]):
            print('video: ', i.split('/')[-1])
            video = np.load(i)
            video = np.expand_dims(video, axis=0)
            #print(features.shape)
            # Prediction of latent vectors for decoder step
            if args.rUnit == 'LSTM':
                enc_outputs_low, low_hidden, low_cell, enc_outputs_high, high_hidden, high_cell = encoder_model.predict(video)
                states_value_first_layer_low = [low_hidden,low_cell]
                states_value_first_layer_high = [high_hidden,high_cell]
                target_seq = np.zeros((1, 1))
                # Populate the first character of target sequence with the start character.
                target_seq[0, 0] = word2int['<bos>']
                stop_condition = False
                decoded_sentence = list()
                alignment_high_steps = list()
                while not stop_condition:
                    output, dec_low_h, dec_low_c, dec_high_h, dec_high_c, alignment_high = decoder_model.predict([enc_outputs_low,states_value_first_layer_low[0],states_value_first_layer_low[1],enc_outputs_high,states_value_first_layer_high[0],states_value_first_layer_high[1],target_seq])
                    # Sample a token
                    sampled_token_index = np.argmax(output[0, -1, :])
                    sampled_char = int2word[sampled_token_index]
                    print('output,',sampled_char)
                    decoded_sentence.append(sampled_char)
                    alignment_high_steps.append(alignment_high)
                    # Exit condition: either hit max length
                    # or find stop character.
                    if (sampled_char == '<eos>' or len(decoded_sentence) > args.Ty):
                        stop_condition = True
                    if search=="Greedy":
                        target_seq = np.zeros((1, 1))
                        target_seq[0, 0] = sampled_token_index
                    states_value_first_layer_low = [dec_low_h, dec_low_c]
                    states_value_first_layer_high = [dec_high_h, dec_high_c]
                
                # Save translated sentence and aligments    
                results[key][i.split('/')[-1]] = [decoded_sentence, alignment_high_steps]
                print(" ".join(decoded_sentence))
                decoded_sentence = list() 
                
            elif args.rUnit == 'GRU':
                pass
            
    with open(args.path2save+name+'.pkl', 'wb') as file:
            pickle.dump(results, file)        
    return results


def video2textBottom(model, encoder_model, decoder_model, args, partition, name='Bottom', search='Greedy'):
    """
    Translation for training and test data.
    """
    # Dics
    word2int = open(args.dataNLP+"dicts/word2int.pkl","rb")
    word2int = pickle.load(word2int)
    int2word = open(args.dataNLP+"dicts/int2word.pkl","rb")
    int2word = pickle.load(int2word) 
    results = defaultdict(dict)
    
    for key in partition.keys():
        print(key, ' data')
        for j, i in enumerate(partition[key]):
            print('video: ', i.split('/')[-1])
            video = np.load(i)
            video = np.expand_dims(video, axis=0)
            #print(features.shape)
            # Prediction of latent vectors for decoder step
            if args.rUnit == 'LSTM':
                enc_outputs_low, low_hidden, low_cell, enc_outputs_high, high_hidden, high_cell = encoder_model.predict(video)
                states_value_first_layer_low = [low_hidden,low_cell]
                states_value_first_layer_high = [high_hidden,high_cell]
                target_seq = np.zeros((1, 1))
                # Populate the first character of target sequence with the start character.
                target_seq[0, 0] = word2int['<bos>']
                stop_condition = False
                decoded_sentence = list()
                alignment_low_steps = list()
                while not stop_condition:
                    output, dec_low_h, dec_low_c, alignment_low, dec_high_h, dec_high_c = decoder_model.predict([enc_outputs_low,states_value_first_layer_low[0],states_value_first_layer_low[1],enc_outputs_high,states_value_first_layer_high[0],states_value_first_layer_high[1],target_seq])
                    # Sample a token
                    sampled_token_index = np.argmax(output[0, -1, :])
                    sampled_char = int2word[sampled_token_index]
                    print('output,',sampled_char)
                    decoded_sentence.append(sampled_char)
                    alignment_low_steps.append(alignment_low)
                    # Exit condition: either hit max length
                    # or find stop character.
                    if (sampled_char == '<eos>' or len(decoded_sentence) > args.Ty):
                        stop_condition = True
                    if search=="Greedy":
                        target_seq = np.zeros((1, 1))
                        target_seq[0, 0] = sampled_token_index
                    states_value_first_layer_low = [dec_low_h, dec_low_c]
                    states_value_first_layer_high = [dec_high_h, dec_high_c]
                
                # Save translated sentence and aligments    
                results[key][i.split('/')[-1]] = [decoded_sentence, alignment_low_steps]
                print(" ".join(decoded_sentence))
                decoded_sentence = list() 
                
            elif args.rUnit == 'GRU':
                pass
            
    with open(args.path2save+name+'.pkl', 'wb') as file:
            pickle.dump(results, file)        
    return results

def evaluation(result, args):
    ann = pd.read_csv(args.annotations+'annotations.csv')
    for data in result.keys():
        reference = ""
        translation = ""
        wert = 0.0
        bleut = 0.0
        meteort = 0.0
        acct = 0.0
        for c, name in enumerate(result[data].keys()):
            translation = result[data][name][0]
            videoName = name.split('.')[0][:11]
            reference = ann['Translation'][ann['Video Name']==videoName].values[0].lower()
            acct = acct + accuracy(copy.deepcopy(reference).split(), copy.deepcopy(translation), args.Ty)
            if '<eos>' in translation:
                translation.remove('<eos>')
            translation = " ".join(translation)
            wert = wert + jiwer.wer(truth = reference, hypothesis = translation)
            bleut = bleut + sentence_bleu([reference], translation, weights=(0, 0, 1, 0))
            meteort = meteort + single_meteor_score(reference, translation)
        print(data)
        print('Acc: ',acct/c)
        print('Bleu: ',bleut/c) 
        print('WER: ',wert/c)
        print('Meteor: ',meteort/c)
        
def evaluationV1(dicts, args):
    references_corpus = list()
    translations_corpus = list()
    ann = pd.read_csv(args.annotations+'annotations.csv')
    for data in results.keys(): 
        for name in results[data].keys():
            videoName = name.split('.')[0][:11]
            references_corpus.append(ann['Translation'][ann['Video Name']==videoName].values[0])
            translations_corpus.append(results[data][name][0][0])
        print(data, ' WER ', jiwer.wer(truth = references_corpus, hypothesis = translations_corpus, words_to_filter=['<eos>']))
        print(data, ' BLEU ', bleu.compute_bleu(reference_corpus=[remove_eos(i.split()) for i in references_corpus], 
                                      translation_corpus=[remove_eos(i.split()) for i in translations_corpus],
                                      max_order=2))
        print(data, ' ROUGE ', rouge.rouge_n(reference_sentences=[remove_eos(i) for i in references_corpus], 
                                  evaluated_sentences=[remove_eos(i) for i in translations_corpus],
                                  n=2))    
        references_corpus = list()
        translations_corpus = list()        