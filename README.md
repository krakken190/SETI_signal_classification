# SETI_signal_classification
Classifying signals from SETI  data

1️⃣  Importing Required Libraries:
Import necessary libraries including pandas, numpy, matplotlib, sklearn, livelossplot, and TensorFlow.

2️⃣  Load and Pre-process the Dataset:
Use pandas to read CSV files containing training and validation data.
Reshape images to the desired format (e.g., (3200, 64, 128, 1)) and convert labels to arrays.

3️⃣  Visualize the Dataset:
Utilize matplotlib to visualize sample images from the dataset.

4️⃣  Create Data Generators:
Create data augmentation generators using ImageDataGenerator from TensorFlow.

5️⃣  Design a Convolutional Neural Network (CNN) Model:
Build a sequential model using Sequential from TensorFlow Keras.
Add Conv2D layers with activation functions and MaxPooling layers.
Incorporate BatchNormalization and Dropout to improve model performance.
Add Flatten layers to prepare data for dense layers.
Create fully connected dense layers with appropriate activation functions.
Finalize the model with an output layer having the softmax activation function.

6️⃣  Compile the Model:
Define the learning rate schedule using ExponentialDecay from TensorFlow.
Compile the model using the Adam optimizer and categorical cross-entropy loss.
Specify accuracy as a metric for monitoring.

7️⃣  Train the Model:
Use ModelCheckpoint to save model weights during training.
Train the model using the fit function and the training data generator.
Monitor validation data using a validation data generator.
Set the number of epochs and include callbacks for live loss plotting.

8️⃣  Evaluate the Model:
Evaluate the trained model using the validation dataset.
Obtain accuracy and loss values using the evaluate method.
Calculate predictions and true labels for generating classification reports.
Display a classification report and confusion matrix using classification_report and confusion_matrix from sklearn.
