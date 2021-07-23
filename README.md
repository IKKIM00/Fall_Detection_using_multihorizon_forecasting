# General Description
A fall detection method with multi-horizon forecasting usring Temporal Fusion Transformers and other deep neural network methods.


# Download Dataset
For `SmartFall` and `Notch` dataset, I have uploaded zip files in `dataset/`. You can also download data through the link below.

### SmartFall and Notch dataset
dataset url - https://userweb.cs.txstate.edu/~hn12/data/SmartFallDataSet/

paper - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6210545/


### DLR dataset
dataset url - https://www.dlr.de/kn/en/desktopdefault.aspx/tabid-12705/22182_read-50785/

### MobiAct dataset
dataset url - https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/


# How to use

## Data Preprocess

Please download all datasets through the URL or `zip` files provided in `dataset/`.

### SmartFall Dataset
1. Put dataset into directory named `dataset/SmartFall_Dataset/`.
2. For SmartFall dataset, preprocessing codes are included in `ipynb` files. 

### Notch Dataset
1. Put dataset into directory named `dataset/Notch_Dataset/`
2. For Notch dataset, preprocessing codes are included in `ipynb` files.

### DLR Dataset
1. Put dataset into directory named `dataset/ARS DLR Data Set/`.
2. Use `dataset/DLR_preprocess.ipynb` for preprocessing. Run all cells in the ipynb file. 
3. Save all preprocessed files in `dataset/dlr_preprocessed`. 

### MobiAct Dataset
1. Put dataset into directory named `dataset/MobiAct_Dataset_v2.0`.
2. Use `dataset/MobiAct_preprocess.ipynb` for preprocessing. Run all cells in the ipynb file.
3. Save all preprocessed files in `dataset/mobiact_preprocessed`.

## For DL Methods
Each file named as `DLR.ipynb`, `MobiAct.ipynb`, `Notch.ipynb`, `SmartFall.ipynb` is for deep learning methods.

In each ipynb file, you can choose which DL method you want to use(`CNN, singleLSTM, stackedLSTM`)


## For TFT Method