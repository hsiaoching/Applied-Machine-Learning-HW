import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()
with open('./data/attributes_list.txt') as f:
    attr_texts = [attr_text[1:-2].replace('/', ' ').split() for attr_text in f]
    for i in range(len(attr_texts)):
        attr_texts[i] = [wordnet_lemmatizer.lemmatize(w, 'v') for w in attr_texts[i]]

def load_attributes(file_name):
    with open(file_name) as f:
        attrs = [map(int, row.split()[1].split(',')) for row in f]
    return np.array(attrs, dtype=int)

def load_attribute_texts(attrs):
    texts = []
    for attr in attrs:
        text = []
        for i in range(len(attr)):
            if attr[i] == 1:
                text.extend(attr_texts[i])
        
        texts.append(set(text))
    return texts

train_feat_alexnet = np.load('./data/alexnet_feat_train.npy')
train_feat_sift = np.load('./data/SIFTBoW_train.npy')
train_feat_attr = load_attributes('./data/attributes_train.txt')
train_raw_attr = load_attribute_texts(train_feat_attr)

test_feat_alexnet = np.load('./data/alexnet_feat_test.npy')
test_feat_sift = np.load('./data/SIFTBoW_test.npy')
test_feat_attr = load_attributes('./data/attributes_test.txt')
test_raw_attr = load_attribute_texts(test_feat_attr)


with open('./data/train.txt') as data:
    labels = list(set([row.split()[1] for row in data]))
    labels_dict = {labels[i]:i for i in range(len(labels))}

with open('./data/train.txt') as data:    
    train_target = np.array([labels_dict[row.split()[1]] for row in data]).astype(int)


ten_k_feat_alexnet = np.load('./data/alexnet_feat_10k.npy')
ten_k_feat_sift = np.load('./data/SIFTBoW_10k.npy')

from json import load
with open('./data/captions.json', 'rb') as caption_file:
    ten_k_captions = load(caption_file)

with open('./data/attributes_list.txt') as data:
    attributes = [row for row in data]


