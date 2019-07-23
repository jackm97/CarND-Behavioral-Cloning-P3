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

[image1]: ./examples/arc_vis.JPG "Model Visualization"
[image2]: ./examples/center.jpg "Center Image"
[image3]: ./examples/recov1.jpg "Recovery Image"
[image4]: ./examples/recov2.jpg "Recovery Image"
[image5]: ./examples/recov3.jpg "Recovery Image"
[image6]: ./examples/data_dist.JPG "Data Distribution"
[image7]: ./examples/downsampled.JPG "Data Distribution"

---
### Files Submitted & Code Quality

#### 1. Submission Files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_best.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_best.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model follows the PilotNet architecture by NVIDIA with some minor differences.  My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 71-96). 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer and Tensorflow's per image standardization method (model.py line 73). The images were cropped so that the environment surrounding the road was out of frame (model.py line 72).

#### 2. Attempts to reduce overfitting in the model
The images from the training dataset were randomly rotated from a normal distribution and randomly flipped with equal probability of being flipped to augment the data and reduce overfitting (lines 43-50).

The model contains a dropout layer after the last convolution in order to reduce overfitting (model.py lines 89). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code lines 108-112). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 98).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and taking extra data in areas the model had difficulty with. 

I also downsampled the data with extremely high occurrence to facilitate a better spread of steering angles. I downsampled the data in an iPython notebook:

![alt text][image6] ![alt text][image7]

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the PilotNet Architecture defined by Nvidia. I thought this model might be appropriate because it was designed as a computationally efficienty architecture for self-driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model seemed to be training well with low mean-squared errors for both the training and validation sets however wasn't producing any decent track results. 

After reducing the dropout layers to just one after the final convolution, results imporoved. However, the simulator still suffered in some specific areas.

Then I gathered more data in these difficult areas. And ran a final test on the track and was able to get around the track without any crashes but there were some recoveries as the car deviated from center a few times. The results can be found in the best_run directory.

#### 2. Final Model Architecture

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps each way on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer away from the edges:

![alt text][image3]

![alt text][image4]

![alt text][image5]

I also got one lap around track 2 to get more extreme turning data.

To augment the data sat, I randomly flipped and rotated training images in a generator passed to Keras's fit_generator method.

After the collection process, I had 48795 number of data points. I then preprocessed this data by cropping and normalizing the images.

I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was determined by the EarlyStopping callback in Keras. A typical number of epochs was 5-10. I used an adam optimizer so that manually training the learning rate wasn't necessary.
