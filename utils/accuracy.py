import copy

def accuracy(reference_, translation_, ty=10):
    
    reference_copy = copy.copy(reference_)
    translation_copy = copy.copy(translation_)
    for j in range(ty - len(reference_)):
        if j==0:
            reference_copy.append('<eos>')
        else:    
            reference_copy.append('<pad>')
    for l in range(ty - len(translation_)):
        translation_copy.append('<pad>')      
    acc = 0.0
    for i in range(ty):
        if reference_copy[i]==translation_copy[i] and reference_copy[i]!='<pad>':
            acc = acc + 1.0
      
    acc = acc/float(len(reference_))
    return acc 