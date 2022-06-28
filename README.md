# CCNet-for-PD-Handwriting
This is the companion repository for our paper titled "Early diagnosis of Parkinson’s disease using Continuous Convolution Network: Handwriting recognition based on off-line hand drawing without template" published in Journal of Biomedical Informatics. 

![architecture cc-net](https://github.com/lizhu1126/CCNet-for-PD-Handwritting/model.jpg)

## Data 
The data used to support the findings of this study are available from the corresponding author upon request.

## Code 
The code is divided as follows: 
* The [train.py] python file contains the necessary code to run an experiement. 
* The [test.py] python file is used to evaluate the experiement. 
* The [cc_net_model.py] python file contains the model. 
* The [data] folder contains the the datasets.
* The [pth] folder contains the best model of the experiement. 

To run a model on one dataset you should issue the following command: 
```
python3 train.py 
python3 test.py
```
## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/dl-4-tsc/blob/master/utils/pip-requirements.txt) file and can be installed simply using the pip command. 
The code now uses Pytorch 1.9.0.


## Acknowledgement

This work was supported by the Zhejiang Provincial Key Lab of Equipment Electronics, Hangzhou, China. This project was also supported by the grants from Zhejiang Health New Technology Product Research and Development Project (2021PY034), Zhejiang Medicine and Health Science and Technology Project (2021KY420) and Medical health Science and Technology project of Zhejiang (2020366842).