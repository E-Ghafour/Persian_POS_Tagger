# Persian POS Tagger

This repository contains implementations of a Persian Part-of-Speech (POS) tagger using different approaches. The implemented models include PyTorch-based model, SpaCy, Average Perceptron, and Python-crfsuite.

## Introduction

Persian POS tagging is a crucial task in natural language processing for understanding the grammatical structure of Persian sentences. This repository provides multiple implementations of a Persian POS tagger using various approaches, allowing users to experiment with different models and select the one that best suits their needs.

![sample output of POS Tagger](pos_tagger.png)

## Approaches

### 1. PyTorch-based POS Tagger

The PyTorch-based POS tagger is implemented using a deep learning model built with PyTorch. It includes word embedding layer with LSTM for training to learn and predict POS tags for Persian words.

### 2. SpaCy POS Tagger

The SpaCy POS tagger utilizes the SpaCy library, and use some prepared structures for training taggers like POS tagger. I use both multilingual bert for embedding layer. This approach provides an efficient and accurate solution for POS tagging.

### 3. Average Perceptron POS Tagger

The Average Perceptron POS tagger is implemented using the average perceptron algorithm of [this repository](https://github.com/sloria/textblob-aptagger/tree/master), a simple, light and effective approach for sequence labeling tasks such as POS tagging.

### 4. Python-crfsuite POS Tagger

The Python-crfsuite POS tagger uses the python-crfsuite library, which is a Python binding for CRFsuite. Conditional Random Fields (CRFs) are employed for sequence labeling tasks like POS tagging. It includes Python-crfsuite and Sklearn libraries for training POS tagger. 

## Selection Process:

In our pursuit of finding the most appropriate approach for POS tagging in [Hazm](https://github.com/roshan-research/hazm), we evaluated different methods based on criteria such as accuracy, resource efficiency, and speed. Our goal was to strike a balance between achieving decent accuracy and ensuring a lightweight and fast solution.

After thorough evaluation, the Python-crfsuite POS tagger emerged as the most suitable choice. It demonstrated commendable accuracy while maintaining a super light and fast performance, aligning perfectly with our criteria. Therefore, Python-crfsuite has been selected as the POS tagging approach for the Hazm library.

Feel free to explore the chosen approach in the `Python_crfsuite` directory for implementation details.

I offer a [pre-trained model](https://drive.google.com/file/d/1Q3JK4NVUC2t5QT63aDiVrCRBV225E_B3/view?usp=sharing) for the Python-crfsuite POS tagger, allowing you to start your POS tagging tasks without the need for training
