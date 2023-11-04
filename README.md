# How important is motion in Sign Language Translation?

Code for "How important is motion in sign language translation?" published by IET Computer Vision

## Graphical Abstract:

<img src="https://drive.google.com/uc?export=download&id=1Jkzqr2dO34kR5QwkySkoGxARXlxOd82r"
     alt="Graphical Abstract"
     style="float: left; margin-right: 10px;" />

## General Pipeline:
<img src="https://drive.google.com/uc?export=download&id=17Iq_xA-LxliWKEfbr1MAN-ygbpybSeCA"
     alt="General Pipeline"
     style="float: left; margin-right: 10px;" />

## Feature extractor:     
<img src="https://drive.google.com/uc?export=download&id=1RULTR0xFVdtvZIvONo4i1sRlf6MJ6Ont"
     alt="Feature extractor"
     style="float: left; margin-right: 10px;" />     

## Results:
<img src="https://drive.google.com/uc?export=download&id=1eXgQB9ngm8BXCg-Zcj7vz9lan9RAK86O"
     alt="Colombian results"
     style="float: left; margin-right: 10px;" />
     
<img src="https://drive.google.com/uc?export=download&id=1xtdSrQ27y_Na3uvcdnr2_bbcRNVs_hON"
     alt="Attention maps"
     style="float: left; margin-right: 10px;" />


## File description:

[setup2.py](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/setup2.py) --> This is the file we use to process the original dataset videos. It applies subsampling, oversampling and flipping operations to the original frames and stores them in the results folder as .npy objects

[mainp.py](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/mainp.py) --> This file calls the files that create the model specified in the Notebook, creates the generators and launches the training.

[NB_SLT_RUN.ipynb](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/NB_SLT_RUN.ipynb) --> Notebook to run the entire proposed approach.

[Performance_Evaluation_Notebook.ipynb](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/Performance_Evaluation_Notebook.ipynb) --> This Notebook evaluates the translations saved with the metrics provided in [NSLT.](https://github.com/neccam/nslt/tree/master/nslt/scripts)

[/utils/augmentation.py](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/utils/augmentation.py) --> This file contains the frame-sampling functions used to augment the Colombian data. 

[/utils/generators.py](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/utils/generators.py) --> This file contains the data generators used to train the models.

[/utils/utils.py](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/utils/utils.py) --> This file contains important functions such as the oversampling and subsampling functions used to augment the data.

[/models/languageModels/translationModels.py](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/models/languageModels/translationModels.py) --> This file contains and creates the different translation models depending on the selected parameters. It also contains the feature extraction network (based on the LTC) and the attention modules used.

[/models/learningModels/learningKerasModels.py](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/blob/master/models/learningModels/learningKerasModels.py) --> This file contains the function that training the created model.

[/results/dataTrain_phoenix_sentences/dicts/](https://github.com/JotaRodriguez94/A_dense_motion_representation_for_attention-based_sign_language_translation/tree/master/results/dataTrain_phoenix_sentences/dicts) --> This folder contains the dictionaries used to create the OneHot Encoding vectors for each video.   


[Download training weights and translations results](https://drive.google.com/drive/folders/1xsmKOyRb6xhIVJzUIPbePqbUjXAoEp87?usp=sharing)

---    
Work done during my master's degree in computer science at the Universidad Industrial de Santander - Colombia under the research group "Biomedical Imaging, Vision and Learning Laboratory". This project performs video translation of Colombian Sign Language and German Sing Language. 
