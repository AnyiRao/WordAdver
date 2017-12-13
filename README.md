## Word-level Adversarial Examples in Convolutional Neural Networks for Sentence Classification

This repository includes robustness improvements using adversarial training improvement. 

### Requirements
Code is written in Python (2.7) and requires Theano (0.9), NLTK.

Using the pre-trained `word2vec` vectors will also require downloading the binary file from
https://code.google.com/p/word2vec/


### Data Preprocessing
To process the raw data, please refer to [https://github.com/AnyiRao/SentDataPre](https://github.com/AnyiRao/SentDataPre)

where path points to the word2vec binary file (i.e. `GoogleNews-vectors-negative300.bin` file). 
This will create a pickle object called `mr.p` in the same folder, which contains the dataset
in the right format.

Note: This will create the dataset with different fold-assignments than was used in the paper.
You should still be getting a CV score of >81% with CNN-nonstatic model, though.

### Version of Codes
``add_one_word.py`` Only one word flip 

``add_two_word.py`` Allow two word flip 

``change_labels.py`` Only one word flip with labels change

``use_pretrained_gene_testset.py`` Use pretrained model to generate adversarial test set and test accuracy (with confidence) on it. 

``conv_net_sentence.pu`` As same as Kim's CNN
### Running the models (CPU)
Example commands:

```
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -rand
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -static -word2vec
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```

This will run the CNN-rand, CNN-static, and CNN-nonstatic models respectively.

### Using the GPU
GPU will result in a good 10x to 20x speed-up, so it is highly recommended. 
To use the GPU, simply change `device=cpu` to `device=gpu` (or whichever gpu you are using).
For example:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec
```
