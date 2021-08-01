import cv2
import numpy as np
import json
from Functions import feature_extractor
from sklearn.svm import SVC
from sklearn.metrics import classification_report as clf_report, confusion_matrix as cm
import joblib
import os


# Global Variables
LABLES = {'1': 'Neutrophil',
          '2': 'Lymphocyte',
          '3': 'Monocyte',
          '4': 'Eosinophil',
          '5': 'Basophil'
          }
DATASETS = ['Raabin-WBC',
            'LISC',
            'BCCD'
            ]


def results_on_dataset(dataset):
    dataset_folder = dataset+'/'
    train_folder, test_folder = 'Train_Aug/', 'Test/'
    os.makedirs(name=dataset_folder+'Features/ALL features/', exist_ok=True)    # making a folder for saving features

    Xs, Ys = [], []     # two empty lists for holding x_train, y_train, x_test, y_test
    for folder in [train_folder, test_folder]:
        # Loading labels from memory (*.json)
        with open(dataset_folder + folder[:-1] + '.json') as file:
            labels = json.load(file)

        # creating a folder for saving each image's feature vector
        os.makedirs(name='%sFeatures/%s'%(dataset_folder, folder), exist_ok=True)

        x, y = [], []
        for name in labels:
            path = dataset_folder + folder + name
            lbl = int(labels[name])
            print(folder[:-1] + '\t' + name + ' -->label: ', str(lbl) + ' = ' + LABLES[str(lbl)])
            img = cv2.imread(path)  # loading image
            # Extracting shape and color features from image
            ncl_detect, error, ftrs = feature_extractor(img=img, min_area=100)

            # saving image feature vector to memory and appending to x and y lists
            if ncl_detect:
                np.save('%sFeatures/%s%s'%(dataset_folder, folder, name), ftrs)
                x.append(ftrs)
                y.append(lbl)
            else:
                f = open('%sFeatures/%s.txt'%(dataset_folder, folder[:-1]), 'a+')
                f.write(name + '\t' + error + '\n')
                f.close()

        Xs.append(np.array(x)), Ys.append(np.array(y))  # appending training and testing features to Xs, Ys

        # Saving x_train, y_train, x_test and y_test in memory
        if folder == 'Train_Aug/':
            np.save(dataset_folder + 'Features/ALL features/x_train', Xs[-1])
            np.save(dataset_folder + 'Features/ALL features/y_train', Ys[-1])
        else:
            np.save(dataset_folder + 'Features/ALL features/x_test', Xs[-1])
            np.save(dataset_folder + 'Features/ALL features/y_test', Ys[-1])

    # Normalizing features using max-min way
    x_train, y_train, x_test, y_test = Xs[0], Ys[0], Xs[1], Ys[1]
    mn, mx = x_train.min(axis=0), x_train.max(axis=0)
    x_train = (x_train - mn)/(mx - mn)
    x_test = (x_test - mn)/(mx - mn)

    # Fitting svm model with best hyperparameters on train set
    clf = SVC(C=6, kernel='poly', class_weight={1: 10}).fit(x_train, y_train)

    # Predicting cells in train and test sets
    train_prds, test_prds = clf.predict(x_train), clf.predict(x_test)
    # Performance of the model on train set
    train_prfmnc = clf_report(y_train, train_prds, digits=4)
    # Performance of the model on test set
    test_prfmnc = clf_report(y_test, test_prds, digits=4)
    # confusion matrix for train set
    train_cm = cm(y_train, train_prds)
    # Confusion matrix for test set
    test_cm = cm(y_test, test_prds)
    # Printing the obtained results
    print('*'*200)
    print('For %s:\nconfusion matrix for train set:\n%s\nclaasification report for train set:\n'
          '%s\nconfusion matrix for test set:\n%s\nclassification report for test set:\n%s'
          %(dataset_folder[:-1], train_cm, train_prfmnc, test_cm, test_prfmnc))

    return {'train_report': train_prfmnc, 'train_cm': train_cm, 'test_report': test_prfmnc, 'test_cm': test_cm}


if __name__ == "__main__":
    dataset_number = input('1:Raabin-WBC  2:LISC  3:BCCD\nPlease choose one of the above datasets: ')
    res = results_on_dataset(DATASETS[int(dataset_number)-1])


