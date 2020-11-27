Compsys 302 AI project group 9
=============================

This project contains our training script, and the 4 neural network models we experimented with to train against our dataset. The dataset trains off the CelebA dataset, which is automatically downloaded when the script is run(if not already downloaded). 

Usage
===========
The model will run by default our chosen model, AlexNet, with the hyperparameters we have selected. Other models can be used by changing the model_num variable on line 91: 1 for VGG-16, 3 for ResNet-50, 4 for LeNet. All the hyperparameters will be automatically adjusted based on the model to provide meaningful results. 

The program will use CUDA if avaliable. 

The CelebA dataset is hosted on google drive, and can sometimes fail to download due to meeting its daily download quota. The only workaround is to download manually from another source to the same path or wait for the quota to refresh (if you are marking this and can't download it contact us and we can try to upload our saved copies for you.)
