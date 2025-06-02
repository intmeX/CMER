import os
import cv2
import random
import numpy as np


def main(data_path=r'C:\koe\DataCenter\emotic_npy', rand_seed=4):
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    test_context = np.load(os.path.join(data_path, 'test_context_arr.npy'))
    test_body = np.load(os.path.join(data_path, 'test_body_arr.npy'))
    test_face = np.stack((np.load(os.path.join(data_path, 'test_face_arr.npy')),) * 3, axis=-1)
    test_cat = np.load(os.path.join(data_path, 'test_cat_arr.npy'))
    num_samples = test_face.shape[0]
    random.seed(rand_seed)
    ids = random.sample(range(num_samples), 10)
    for idx, i in enumerate(ids):
        context = cv2.cvtColor(test_context[i], cv2.COLOR_RGB2BGR)
        body = cv2.cvtColor(test_body[i], cv2.COLOR_RGB2BGR)
        face = cv2.cvtColor(test_face[i], cv2.COLOR_RGB2BGR)
        label = test_cat[i]
        cv2.imwrite(os.path.join('context{}.png'.format(idx)), context)
        cv2.imwrite(os.path.join('body{}.png'.format(idx)), body)
        cv2.imwrite(os.path.join('face{}.png'.format(idx)), face)
        with open('label{}.txt'.format(idx), 'a', encoding='utf-8') as f:
            for j in range(len(label)):
                if label[j]:
                    f.write(cat[j] + '\n')
        # cv2.imshow('', context)
        # cv2.waitKey(0)
        # cv2.imshow('', body)
        # cv2.waitKey(0)
        # cv2.imshow('', face)
        # cv2.waitKey(0)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
