### Convolutional-Neural-Networks - Image-Classification 
This repository containing CNN - Image classification projects was implemented by me. 
It was programmed in Google colaboratory (a cloud-based data analysis and machine learning tool 
that allows you to combine executable Python code and rich text along with charts, images, HTML, 
LaTeX and more into a single document stored in Google Drive.)

**Convolutional Neural Network, also known as convnets or CNN**, is a well-known method in computer vision applications. 
It is a class of deep neural networks that are used to analyze visual imagery. 

This type of architecture is dominant to recognize objects from a picture or video. 
It is used in applications like image or video recognition, neural language processing, etc. 

For example Facebook uses CNN for automatic tagging algorithms, 
Amazon — for generating product recommendations and 
Google — for search through among users’ photos. 

![My Image](https://miro.medium.com/max/1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg) 

In this repository, we have used CNN to classify Emergency from Non Emergency Vehicles, 
Detect and classify blood cell images with malaria parasites, and classify flowers into 5 
different classes. 

The dataset used in this repository was gotten from KAGGLE and TENSORFLOW
[Emergency from Non Emergency Vehicles](https://www.kaggle.com/datasets/kishor1210/emergency-vs-nonemergency-vehicle-classification), 
[Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria), 
[Flower photos](http://download.tensorflow.org/example_images/flower_photos.tgz). 

This project was completed using the following Tools and Frameworks 

**Tools | Frameworks** 
- **Data Preprocessing:** NumPy, Pandas, OpenCV 
- **Deep Learning Implementation Framework:** Scikit-Learn, TensorFlow, Keras 
- **Data Visualization:** Matplotlib, Seaborn. 
- **Integrated Development Environment:** Google colaboratory
- **Version control system:** Git and Github 

### CNN - Image classification pipeline 

![My Image](https://www.datanami.com/wp-content/uploads/2021/04/pipeline_shutterstock_Aurora72.jpg) 

- 1 Download, Upload and Unzip the image dataset 
The image dataset was downloaded from [KAGGLE](https://www.kaggle.com/), uploaded to my Google drive and then Unzipped with Python **ZipFile** module in Google Colaboratory.
- **2 Read the image data file directory:** 
I read the image data file directory with the pathlib module. 

- **3 Visualize the images:** 
Matplotlib was used to visualize the images
- **4 Image Data Preprocessing:**
	  a. Convert to numpy arrays 
	  b. Normalize (scale) the arrays 
- **5 Split the image dataset into train, validation and test data:** 
Data splitting is the act of partitioning available data into three portions, usually for cross-validatory purposes. The training and validation portion of the data is used to develop a predictive model, and the test portion to evaluate the model's performance. Scikit-learn train_test_split function was used to split the data into training, validation, and testing set 

- **6 Data Augmentation:** 
Data augmentation is a process of artificially increasing the amount of data by generating new data points from existing data. This includes adding minor alterations to data like resizing, flipping, rotating, cropping, padding, etc. This process was executed with the image.preprocessing class of TensorFlow 

- **7 Train the Model:** 
A deep learning training model is a process in which a deep learning algorithm is fed with sufficient training data to learn from. The model was train sequentially stacking layers of Neural Networks where each layer has exactly one input tensor and one output tensor. 

- **8 Model Evaluation:** The process of using different evaluation metrics to understand a deep learning model's performance, as well as its strengths and weaknesses. Model evaluation is important to assess the efficacy of a model during initial research phases, and it also plays a role in model monitoring

- **9 Save the Model:** Saving the  model architecture, learned weights to permits their reusage in the future or in production.


