# Fingerprint_Verification_Encoder
This is part of a team project for the Integrated Design Project courseware.
Our project title is: Biometric-Secured Electronic Health Record (EHR) System

My tasks include:
1. Training of a Siamese Convolutional Neural Network for fingerprint verification system
2. Integrate the fingerprint verification system into our EHR web application

To train a fingerprint verification encoder via Siamese/Triplet network.
<br>The dataset used is [Kaggle: Sokoto Coventry Fingerprint Dataset (SOCOFing)](https://www.kaggle.com/ruizgara/socofing)

## Sample Result
Here are examples of the performance on the Validation Dataset (splitted from the original dataset via K-Fold Split)

<img src="https://raw.githubusercontent.com/yjwong1999/Fingerprint_Verification_Encoder/main/others/fingerprint.png" 
     alt="Sample Result" width=700>

## Hardware Setup
- Raspberry Pi 4 Model B - 4GB
- Fingerprint Reader Sensor - R307
- USB to TTL (Serial) Converter Module - PL2303HX

<img src="https://raw.githubusercontent.com/yjwong1999/Fingerprint_Verification_Encoder/main/others/EHR%20Demo.png" 
     alt="Hardware Setup" width=700>
     
## Demo
Kindly click the link below to see the project's demonstration for the fingerprint verifaction part! <br>
[Demonstration Link](https://www.youtube.com/watch?v=z5eb6lPQgg4&t=155s)

## TODO
1) Upload the Jupyter Notebook used to train the neural network
- The notebook is currently edited for a neater view
3) Fine tune the model
- Although the Siamese Convolutional Neural Network perform quite well in the training/validation setup (96-97% accuracy)
- The model is not performing that well using our hardware setup (because the fingerprint sensor used in our project is different from the one used by the Kaggle Dataset)
