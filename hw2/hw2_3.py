import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skimage import io
import os
import random

img_folder = 'Pictures'

class ImageLoader():
    def __init__(self, folder, split_ratio):
        self.img_dict = {i: {'train': [], 'test': []} for i in range(10)}
        for idx in range(10):
            sub_folder = os.path.join(folder, str(idx))
            images = []
            for filename in os.listdir(os.path.join(sub_folder)):
                img = io.imread(os.path.join(sub_folder, filename), as_gray=True)
                images.append(img.reshape(-1))
            image_train, image_test = train_test_split(images, shuffle=True,
                                                       train_size=split_ratio,
                                                       test_size=1-split_ratio) # shuffle and split
            self.img_dict[idx]['train'] = np.array(image_train)
            self.img_dict[idx]['test'] = np.array(image_test)

    def load_image_data(self, sample_size):
        assert sample_size > 0
        sample_idxes = random.sample(range(len(self.img_dict)), sample_size)
        print('id of sample data: ' + str(sample_idxes))
        X_train = np.array(self.img_dict[sample_idxes[0]]['train'])
        Y_train = np.array([sample_idxes[0]] * len(X_train)).reshape(-1, 1)
        X_test = np.array(self.img_dict[sample_idxes[0]]['test'])
        Y_test = np.array([sample_idxes[0]] * len(X_test)).reshape(-1, 1)
        for idx in sample_idxes[1:]:
            X_train = np.vstack((X_train, self.img_dict[idx]['train']))
            Y_train = np.vstack((Y_train, np.array([idx] * len(self.img_dict[idx]['train'])).reshape(-1, 1)))
            X_test = np.vstack((X_test, self.img_dict[idx]['test']))
            Y_test = np.vstack((Y_test, np.array([idx] * len(self.img_dict[idx]['test'])).reshape(-1, 1)))
        return X_train, Y_train.reshape(-1), X_test, Y_test.reshape(-1)

if __name__ == '__main__':
    ''' question 3 '''
    sample_num = [2, 5, 7, 10]
    round = 10
    train_score = {key: [] for key in sample_num}
    test_score = {key: [] for key in sample_num}
    for i in range(round):
        for num in sample_num:
            img_loader = ImageLoader(img_folder, 0.75)
            X_train, Y_train, X_test, Y_test = img_loader.load_image_data(num)
            model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
            model.fit(X_train, Y_train)
            score_temp = [model.score(X_train, Y_train) * 100, model.score(X_test, Y_test) * 100]
            print('accuracy on training data: %.2f%%' % score_temp[0])
            print('accuracy on test data: %.2f%%' % score_temp[1])
            train_score[num].append(score_temp[0])
            test_score[num].append(score_temp[1])
    print('---' * 10)
    print('average result: ')
    for num in sample_num:
        print('sample num: ' + str(num))
        print('training accuracy: %.2f%%' % (sum(train_score[num]) / round))
        print('test accuracy: %.2f%%' % (sum(test_score[num]) / round))
