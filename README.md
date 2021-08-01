
# New Segmentation and Feature Extraction Algorithm for Classification of White Blood Cells in Peripheral Smear Images 

We address a new method for the classification of white blood cells (WBCs) using image processing techniques and machine learning methods. The proposed method consists of three steps: detecting the nucleus and cytoplasm, extracting features, and classification. At first, a new algorithm is designed to segment the nucleus. For the cytoplasm to be detected, only a part of it located inside the convex hull of the nucleus is involved in the process. This attitude helps us overcome the difficulties of segmenting the cytoplasm. In the second phase, three shapes and four novel color features are devised and extracted. Finally, by using an SVM model, the WBCs are classified. The segmentation algorithm can detect the nucleus with a dice similarity coefficient of 0.9675. The proposed method can categorize WBCs in RaabinWBC, LISC, and BCCD datasets with accuracies of 94.65 %, 92.21 %, and 94.20 %, respectively. It is worth mentioning that the hyperparameters of the classifier are fixed only with the Raabin-WBC dataset, and these parameters are not readjusted for LISC and BCCD datasets. The obtained results demonstrate that the proposed method is robust, fast, and accurate. The paper is available at:
https://doi.org/10.1101/2021.04.29.441751


## Requirements
The method is developed in python3 and the following libraries should be installed:

* Python: 3.7
* Numpy: 1.19.1
* opencv-python: 3.4.2.16
* scikit-image: 0.16.2
* scikit-learn: 0.23.2
* scipy: 1.5.2
* pyhdust: 1.3.26

## Steps to setup and execute the code
### Data preparation
We evaluated our work with three datasets which are [Raabin-WBC](https://doi.org/10.1101/2021.05.02.442287) , LISC, and BCCD. 
