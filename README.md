# hand_writing_recognition

This is a final project in the course Pattern Recognition. This is the implementation of the handwritten digits recognition using shape feature extraction and classification using Cosine Similarity and KNN. The dataset used is from the ARDIS dataset. For the full report, please refer to [Pattern Recognition Final Assignment: Handwritten Digits Classification using Shape Features](Pattern_Recognition_Final_Assignment_Report.pdf)

## Introduction

In image processing, feature extraction plays a significant role. Prior to getting the features, the images have went through several preprocessing steps that are taken accordingly to the goal of the image processing. When extracting the features, it is necessary to apply the appropriate techniques.

## Data Acquisition

There are 30 images used for the training data and 5 images used for testing data in each class, 0 to 9. The data are taken from the ARDIS (Arkiv Digital Sweden) dataset. The images in ARDIS dataset are extracted from 15.000 Swedish church records which were written by different priests with various handwriting styles in the nineteenth and twentieth centuries.

## Image Preprocessing

The image preprocessing technique used is binarization, where the handwritten digit images are converted into black and white images. The Otsu method is used to find the pixel value threshold in the image histogram.

## Feature Extraction

The features that are being extracted from the images are the based on the shape and the skeleton of the image. The shape measurements include area, perimeter, and compactness, whereas the skeleton measurement include the number of holes.

## Classification

The classification is done by comparing the calculation of cosine similarity and K-nearest neighbour with K=5. The result shows that KNN gives better accuracy of 34%.

## Conclusion

As opposed to classification using the features of pixel intensity, i.e. color, in classifying 10 handwritten digits, it is more preferred to extract the distinctive features of each digit shape. By extracting the shape features of the digit object, there are new features of the original image that are discovered such as the compactness and the number of holes.

The results of the classification using cosine similarity and KNN for two features, the number of holes and object compactness are 26% and 34%, respectively. The low accuracy result is caused by the different patterns in the handwriting styles in the images. That said, it effects the calculation of the image holes and therefore can make the feature calculation incorrect. Further- more, after trying to add the other two features, i.e. the perimeter and area, the cosine similarity matching shows an increase of 2% after adding each feature, while the KNN classification accuracy remains the same. It can be inferred that adding the area and the perimeter as additional features is treated redundant in the KNN classification.
