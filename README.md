# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: writeup_images/im1.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model includes convolutional layers with 5x5 kernels (code lines 71 - 77) and fully connected layers (code lines 83 - 90). The model includes RELU layers to introduce nonlinearity (code line 71, 76), and the data is normalized in the model using a Keras lambda layer (code line 66). Also, Cropping2D layer was used to select region of interests (code line 68).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py lines 73, 84). The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 93). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia paper provided in lectures. I thought this model might be appropriate because the instructor provided enough materials to be sure that this model will give a satisfactory result for solving the behavioral cloning problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that added dropout layers.

Then I played with number of epochs and stopped on 5. 

Training and Validation Loss Metrics:

![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I retrained model by driven more on this places.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 66-90) consisted of a convolution neural network with the following layers and layer sizes ...

| Layer         		|     Description	        					                | 
|:---------------------:|:---------------------------------------------:                | 
|Input                  | RGB image (160,320,3)                                         |
|Lambda                 | Normalize the image pixels, outputs (None, 160, 320, 3)|                                
|Cropping2D             | Crops the image, outputs (None, 65, 320, 3)|                                
|Convolution2D          | 5X5 kernel, (2,2) sampling, relu activation, outputs(None, 31, 158, 24)    |
|Convolution2D          | 5X5 kernel, (2,2) sampling, relu activation, outputs(None, 14, 77, 36)    |
|Convolution2D          | 5X5 kernel, (2,2) sampling, relu activation, outputs(None, 5, 37, 48)    |
|Convolution2D          | 3X3 kernel, relu activation, outputs(None, 3, 35, 64)    |
|Convolution2D          | 3X3 kernel, relu activation, outputs(None, 1, 33, 64)    |
|Flatten                |outputs (None, 2112)              |
|Dense                  |outputs (None, 100)                |
|Dropout                |outputs (None, 100)                |              
|Dense                  |outputs (None, 50)          |
|Dropout                |outputs (None, 50)                |
|Dense                  | outputs (None, 10)          |
|Dropout                |outputs (None, 10)                |
|Dense                  | (None, 1)            |    

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
