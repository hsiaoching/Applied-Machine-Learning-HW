import numpy as np

train_feat_alexnet = np.load('./data/alexnet_feat_train.npy')
train_feat_sift = np.load('./data/SIFTBoW_train.npy')

with open('./data/train.txt') as data:
    labels = list(set([row.split()[1] for row in data]))
    labels_dict = {labels[i]:i for i in range(len(labels))}

with open('./data/train.txt') as data:    
    train_target = np.array([labels_dict[row.split()[1]] for row in data]).astype(int)

ten_k_feat_alexnet = np.load('./data/alexnet_feat_10k.npy')
ten_k_feat_sift = np.load('./data/SIFTBoW_10k.npy')

