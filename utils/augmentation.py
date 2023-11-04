import numpy as np
from utils.utils import oversampling


def frame_sampling(frames, nTempo):
    # Make sure that len_frames >= nTempo
    len_frames = len(frames)
    if len_frames < nTempo:
        print('Len before', len_frames)
        frames = oversampling(list(frames), mean=nTempo)
        len_frames = len(frames)
        print('Len after', len_frames)
        
    rate = len_frames // nTempo
    mod = len_frames % nTempo
    frames_list = list()
    last = 0
    indexes = list()
    
    if len_frames - rate >= nTempo * rate:  
        # Adjust the size of the step so as not to exceed the number of frames.
        adjusted_rate = (len_frames - rate) / nTempo
        # Calculate the percentage of final steps larger than the rate.
        per = adjusted_rate - rate
        # Calculate the number of initial and final steps
        nFinalSteps = int(nTempo * per) 
        nFirstSteps = nTempo - nFinalSteps
    
    elif mod < rate:
        per = 0
        nFinalSteps = 0
        nFirstSteps = nTempo - nFinalSteps 
        
    print(rate, " clips of ", nTempo)
    print("Each clip has ", nFirstSteps, " steps of ", rate, " and ", nFinalSteps, " steps of ", rate+1)
        
    # Reverse k to keep the sequence
    x = lambda a: a==0
    for i in (range(0,rate)):
        for k, n in enumerate([nFirstSteps, nFinalSteps]):
            # Calculate the end of the range
            end = (last+(n*(rate+k)))+(rate*k)
            # Select indexes, concatenating the different sizes of step.
            indexes = indexes + [j+(i*int(x(k))) for j in range(last,end,rate+k)][1*k:]
            # To couple with the second sequence.
            last = indexes[-1]
        frames_list.append(np.array(frames)[indexes])    
        last = 0
        indexes = list()

    return frames_list

def flip_vertical(volume):
    return np.flip(volume, (0, 2))[::-1]
    
    
    
    
