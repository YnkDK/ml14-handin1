Machine Learning (2014, Q1)
===========================

Implementation of Logistic Regression (One-vs-All) and Softmax for classifying optical character recognition (OCR)

To generate the best parameters for logistic regerssion on mnist data run the MATLAB script src/logTrainMnist.m

To generate the best parameters for logistic regerssion on AU data, download the data at the [Course Homepage](https://services.brics.dk/java/courseadmin/ML14/documents/getDocument/auTrain.mat?d=136530) and place it in dat/, then run the MATLAB script src/logTrainAu.m

Once the AU data is downloaded, the softmax can train both on AU training data and Mnist training data (src/train.m)

The best parameters found in our experiments are contained in src/[soft|log]BestParams.mat
