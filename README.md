# American Sign Language Detection

This project aims to detect American Sign Language symbols using hand landmarks recorded with Mediapipe and an Artificial Neural Network (ANN) model. The project integrates OpenCV to perform real-time hand symbol detection using the trained model.

## Dataset Creation

The dataset was created using Mediapipe, which records hand landmarks while the user performs different ASL symbols. The hand landmarks, along with their corresponding labels (ASL symbols), were stored in a keypoint.csv file.

## Data Preprocessing

For preprocessing the data:

1. Extract the co-ordinates of landmarks (ignoring z co-ordinate) from the mediapipe results.
2. Every landmark is converted considering the origin as the wrist
3. Hand landmarks are then sclaed down to the range 0 - 1 and stored in the file.

## Model Training

An ANN model was trained using the preprocessed dataset. The steps involved in model training include:

1. Define the architecture of the ANN, including the number and type of layers, activation functions, and regularization techniques.
2. Train the model on the training set using appropriate loss functions and optimization algorithms.
3. Optimize the model to improve its performance.
4. You can use the training.ipynb file as refrence.

## Model Evaluation

The trained ANN model was evaluated on the testing set to assess its performance. Metrics such as accuracy, precision, recall, F1 score and heatmap were calculated to measure the model's ability to correctly classify the ASL symbols. For the given Training.ipynb file the accuracy on testing set is 96.64%

## Real-time Hand Symbol Detection

Real-time hand symbol detection is performed by integrating OpenCV with the trained ANN model. The following steps are involved:

1. Access the camera using OpenCV to capture video frames.
2. Use Mediapipe to detect and extract hand landmarks from the video frames.
3. Preprocess the extracted landmarks.
4. Pass the preprocessed landmarks through the trained ANN model for prediction.
5. Store the results of the frame , Take out the maximum occuring symbol out of 200 frames and display it on the video feed
6. Show nothing on camera for a space and use backspace button to delete chars from string.


   <img width="960" alt="Demo-hello" src="https://github.com/MohdDilshad-nitk/ASL_Recognition/assets/97335106/9fa510b0-22b3-43c8-be80-bfbeff4f59db">


## Potential Challenges

- Lighting conditions: Ensure that the hand landmarks are accurately detected and recorded under different lighting conditions. Although Mediapipe works brilliant under different lighting conditions
- Background noise: Minimize the impact of background noise and distractions to improve the accuracy of hand symbol detection.
- Model limitations: Be aware that the trained model may have limitations and may not accurately detect all ASL symbols in every scenario.

## Requirements

- Python 
- OpenCV 
- Mediapipe
- TensorFlow 
- Numpy 
- Pandas 
- Scikit-learn 

## Usage

1. Clone this repository.
2. Install the required dependencies mentioned in the Requirements section.
3. Run the app.py, which will start the real-time hand symbol detection using the camera.
4. To create your own dataset , go to keypoint_classifier folder and empty the keypoint.csv file(note do not delete it, if you delete it create another empty file and place it in the directory). Run app.py and press the `k` key to initialise the data storing mode. Now show the symbol on camera and press the designated alphabet key. The data will be stored after preprocessing along with the label.
5. To edit the labels go to keypoint_classifier folder and edit keypoint_classifier_label.csv file. 

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your improvements.
