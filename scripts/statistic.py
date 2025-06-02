import os
import numpy as np
import torch


def main():
    cat_name = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness',
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat_name):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion
    train_cat = np.load(os.path.join(r'C:\koe\DataCenter\emotic_npy', 'train_cat_arr.npy'))
    val_cat = np.load(os.path.join(r'C:\koe\DataCenter\emotic_npy', 'val_cat_arr.npy'))
    test_cat = np.load(os.path.join(r'C:\koe\DataCenter\emotic_npy', 'test_cat_arr.npy'))
    for cat in [train_cat, val_cat, test_cat]:
        pos_num = torch.tensor(cat.sum(axis=0), dtype=torch.float32) / cat.shape[0] * 100
        info = [[round(pos_num[i].item(), 2), ind2cat[i]] for i in range(26)]
        info.sort(key=lambda x: x[0], reverse=True)
        for item in info:
            print(item[1], ' ', str(item[0]), sep='')
        # pos_weight = (cat.shape[0] - pos_num + 1.0) / (pos_num + 1.0)
        # print('the neg/pos of dataset:', pos_weight)
        # pos_ratio = (pos_num + 1.0) / (cat.shape[0] + 1.0)
        # print('the pos ratio of dataset:', pos_ratio)
    # pos_ratio = torch.tensor(train_cat.sum(axis=0), dtype=torch.float32)
    # info = [[int(pos_ratio[i].item()), ind2cat[i]] for i in range(26)]
    # info.sort(key=lambda x: x[0], reverse=True)
    # for item in info:
    #     print(item[1], ' ', str(item[0]), sep='')
    # pos_num = torch.tensor(train_cat.sum(axis=0), dtype=torch.float32)
    # neg_pos_weight = (train_cat.shape[0] - pos_num + 1.0) / (pos_num + 1.0)
    # info = [[round(pos_ratio[i].item(), 2), ind2cat[i]] for i in range(26)]
    # info.sort(key=lambda x: x[0], reverse=True)


if __name__ == '__main__':
    main()
