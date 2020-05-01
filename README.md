# Spatial Temporal Adversarial Autoencoder
## Description
  Spatial Temporal Adversarial Autoencoder is a model that based on ST-GCN and AAE. The model is a multi task modal tha able to classify the action class, cluster the class of action and generate skeleton data.

## Initialization
### Pre-Installation
  The package for the model can be downlaod and install with the command:
  ```
  pip install --upgrade --force-reinstall -r requirements.txt
  ```
  Beside, torchlight module is used in the code and t can be install with te command:
  ```
  cd ./torchlight ; python setup.py install ; cd ..
  ```
  After That, restart of the terminal seasion is need to apply latest update of the installation
### Dataset 
  NTU RGB+D 120 Dataset ,can be download from URL: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp
  
  For Data Extraction and preprocessing for the model, used the command:
  ```
  pyhon ./data_gen/<type of dataset generator> --path <path of the skeleton data of the dataset>
  # <type of dataset generator> = ntu5_gendata.py   for dataset with 5 action classes
  #                             = ntu20_gendata.py  for dataset with 20 action classes
  #                             = ntu120_gendata.py for dataset with 120 action classes which is the wwhole set
  ```
### Pre-Training model
  The weight of the model can be download from the Drive:
  URL: 
  
## ST-AAE
### Run Model 
  For run supervised learning model :
  ```
  python main_supervised.py   --config   "./config/<dataset config>/train.yaml" 
                              --work_dir <output working directory>
                              --weights  <path to the Model weights file>
  ```
### Display The Skeleton Data
  Display the Skeleton data and Reconstructed skeleton data with commmand:
  ```
  python display_data.py  --data  <path of original data file>
                          --recon <path of the reconstructed data file>
                          --save  <True for save all the figure per frame>
  ```
