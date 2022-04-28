
# New Segmentation and Feature Extraction Algorithm for Classification of White Blood Cells in Peripheral Smear Images 

We address a new method for the classification of white blood cells (WBCs) using image processing techniques and machine learning methods. The proposed method consists of three steps: detecting the nucleus and cytoplasm, extracting features, and classification. At first, a new algorithm is designed to segment the nucleus. For the cytoplasm to be detected, only a part of it located inside the convex hull of the nucleus is involved in the process. This attitude helps us overcome the difficulties of segmenting the cytoplasm. In the second phase, three shapes and four novel color features are devised and extracted. Finally, by using an SVM model, the WBCs are classified. The segmentation algorithm can detect the nucleus with a dice similarity coefficient of 0.9675. The proposed method can categorize WBCs in RaabinWBC, LISC, and BCCD datasets with accuracies of 94.65 %, 92.21 %, and 94.20 %, respectively. It is worth mentioning that the hyperparameters of the classifier are fixed only with the Raabin-WBC dataset, and these parameters are not readjusted for LISC and BCCD datasets. The obtained results demonstrate that the proposed method is robust, fast, and accurate. The paper is available at:
https://doi.org/10.1038/s41598-021-98599-0


## Requirements
The method is developed in python3 and the following libraries should be installed:

* Python: 3.7
* Numpy: 1.19.1
* opencv-python: 3.4.2.16
* scikit-image: 0.16.2
* scikit-learn: 0.23.2
* scipy: 1.5.2
* pyhdust: 1.3.26

## Steps to set up and execute the code
### Step1: Data preparation
We evaluated our work with three datasets [Raabin-WBC](https://www.nature.com/articles/s41598-021-04426-x), [LISC](https://doi.org/10.1016/j.compmedimag.2011.01.003), and BCCD. 
* We cropped the white blood cells of the LISC dataset and made them suit for our own work. So if you use the LISC dataset, you must cite its [paper](https://doi.org/10.1016/j.compmedimag.2011.01.003). Download the cropped images of the LISC dataset from [here](https://drive.google.com/file/d/1gknVrSs1CRy8PoIh1HXiGu-1ObH3cQ9S/view?usp=sharing). Also, you can download the original LISC dataset from [here](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm).
* Besides the LISC dataset, we also used the BCCD dataset. The original BCCD dataset is available from Kaggle. We made this dataset suit for our own work. Download the dataset from [here](https://drive.google.com/file/d/1h-wuDURfuKeJYvKOWTcYpuyMxNy0lzIt/view?usp=sharing).
* Finally, download the Double-labeled Raabin-WBC dataset from [here](https://drive.google.com/file/d/1-aPhQyakD79vKYh2l0fPsT2xCiX3UMYi/view?usp=sharing). Note that these data are the same as [the original version](http://dl.raabindata.com/WBC/Cropped_double_labeled/), except these data have been prepared for this repo and the related paper.
### Step2:
* After downloading the datasets, extract and put them beside the main.py . Then,  you can run the main.py. Type 1 or 2 or 3 to select the dataset.
## Citation
Please cite [the corresponding paper](https://doi.org/10.1038/s41598-021-98599-0) and [Raabin-WBC paper](https://doi.org/10.1038/s41598-021-04426-x). The IEEE style of citations:
* Tavakoli, S., Ghaffari, A., Kouzehkanan, Z.M. et al. New segmentation and feature extraction algorithm for classification of white blood cells in peripheral smear images. Sci Rep 11, 19428 (2021). https://doi.org/10.1038/s41598-021-98599-0
* Kouzehkanan, Z.M., Saghari, S., Tavakoli, S. et al. A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm. Sci Rep 12, 1123 (2022). https://doi.org/10.1038/s41598-021-04426-x





