# General Description
This code is for paper "Fall Detection Based on Multi-Horizon Forecasting"( Currently under review )

A fall detection method with multi-horizon forecasting usring Temporal Fusion Transformers and other deep learning methods.

For other deep learning methods, 1D CNN, single LSTM, stacked LSTM were used.

All models were configured to forecast falls through the window size of data from the perspective of regression instead of classification.

For the last predicted value, the class of the predicted values was classified on the basis of the threshold value. 

To verify benchmark performance, we faithfully reproduced the 1D CNN and LSTM-basedmodels, although the model structures were modified to enable regression in both cases because they were tailored to the classification task.

1D CNN model architecture is based on the model structure proposed in 2020 by Kraft et al([paper](https://github.com/IKKIM00/Fall_Detection_using_multihorizon_forecasting/files/6866631/Deep.Learning.Based.Fall.Detection.Algorithms.for.Embedded.Systems.Smartwatches.and.IoT.Devices.Using.Accelerometers.pdf)).

Signle LSTM and stacked LSTM model is based on the model architecture proposed in 2019 by Luna et al ([paper](https://github.com/IKKIM00/Fall_Detection_using_multihorizon_forecasting/files/6866652/sensors-19-04885-v2.pdf)).

# Requirements
`python==3.7.3`

`tensorflow-gpu==2.5.3` or `tensorflow==2.5.3`

`sklearn==0.24.2`

## Multi-horizon forecasting result
![model_pred](https://user-images.githubusercontent.com/37397258/126738142-7fc1218d-eb55-4f88-9c24-4112b320354b.jpg)
Prediction results for the SmartFall, Notch, DLR and MobiAct datasets in order using:

(a)-(d) TFT method, (e)-(h) Single LSTM, (i)-(l) Stacked LSTM, (m)-(p) 1D CNN

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
1. Clone https://github.com/google-research/google-research/tree/master/tft
2. Use each file named as `dlr_tft.ipynb`, `mobi_tft.ipynb`, `notchFall_tft.ipynb` and `smartFall_tft.ipynb` for TFT method.
3. Files named as `dlr_tft_wo_bioinfo.ipynb` and `mobi_tft_no_bioinfo.ipynb` are for cases when personal biometric information is removed.
