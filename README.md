
# New Segmentation and Feature Extraction Algorithm for Classification of White Blood Cells in Peripheral Smear Images 

We address a new method for the classification of white blood cells (WBCs) using image processing techniques and machine learning methods. The proposed method consists of three steps: detecting the nucleus and cytoplasm, extracting features, and classification. At first, a new algorithm is designed to segment the nucleus. For the cytoplasm to be detected, only a part of it located inside the convex hull of the nucleus is involved in the process. This attitude helps us overcome the difficulties of segmenting the cytoplasm. In the second phase, three shapes and four novel color features are devised and extracted. Finally, by using an SVM model, the WBCs are classified. The segmentation algorithm can detect the nucleus with a dice similarity coefficient of 0.9675. The proposed method can categorize WBCs in RaabinWBC, LISC, and BCCD datasets with accuracies of 94.65 %, 92.21 %, and 94.20 %, respectively. It is worth mentioning that the hyperparameters of the classifier are fixed only with the Raabin-WBC dataset, and these parameters are not readjusted for LISC and BCCD datasets. The obtained results demonstrate that the proposed method is robust, fast, and accurate. The paper is available at:
https://www.nature.com/articles/s41598-021-04426-x


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
### Step1: Data preparation
We evaluated our work with three datasets [Raabin-WBC](https://doi.org/10.1101/2021.05.02.442287), [LISC](https://doi.org/10.1016/j.compmedimag.2011.01.003), and BCCD. 
* We cropped the white blood cells of the LISC dataset and made them suit for our own work. So if you use the LISC dataset, you must cite its [paper](https://doi.org/10.1016/j.compmedimag.2011.01.003). Download the cropped images of the LISC dataset from [here](https://drive.google.com/file/d/1gknVrSs1CRy8PoIh1HXiGu-1ObH3cQ9S/view?usp=sharing). Also, you can download the original LISC dataset from [here](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm).
* Besides the LISC dataset, we also used the BCCD dataset. The original BCCD dataset is available from Kaggle. We made this dataset suit for our own work. Download the dataset from [here](https://drive.google.com/file/d/1h-wuDURfuKeJYvKOWTcYpuyMxNy0lzIt/view?usp=sharing).
* Finally, download the [Raabin-WBC dataset](http://dl.raabindata.com/WBC/Cropped_double_labeled/). 
### Step2:
* After downloading the datasets, extract and put them beside the main.py . Then,  you can run the main.py. Type 1 or 2 or 3 to select the dataset.
## Citation
If you use the Raabin-WBC dataset, you should cite the [corresponding paper](https://doi.org/10.1101/2021.05.02.442287). If you use the ground truths utilized in this paper, please cite [our paper](https://doi.org/10.1101/2021.04.29.441751) in addition to Raabin-WBC paper. The IEEE style of citations:
* Z. M. Kouzehkanan et al., “Raabin-WBC: a large free access dataset of white blood cells from normal peripheral blood,” bioRxiv, p. 2021.05.02.442287, Jan. 2021, doi: 10.1101/2021.05.02.442287.
* S. Tavakoli, A. Ghaffari, Z. M. Kouzehkanan, and R. Hosseini, “New Segmentation and Feature Extraction Algorithm for the Classification of White Blood Cells in Peripheral Smear Images,” bioRxiv, 2021, doi: 10.1101/2021.04.29.441751.





