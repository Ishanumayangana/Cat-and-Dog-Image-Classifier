Cat and Dog Image Classifier

This project is an implementation of a Cat and Dog image classifier using a convolutional neural network (CNN). The dataset used for training and testing is sourced from Kaggle, comprising images of cats and dogs.
Dependencies

    TensorFlow
    Matplotlib
    Pandas

Project Structure

    load_dataset.py: Script to load the dataset.
    scaling_images.py: Script to preprocess and scale the images.
    data_augmentation.py: Script to perform data augmentation.
    model_building.py: Script to construct the CNN model architecture.
    model_training.py: Script to train the model.
    performance_analysis.py: Script to analyze model performance.
    model_evaluation.py: Script to evaluate the trained model.

Steps
1. Load Dataset

The load_dataset.py script loads the Cat and Dog dataset from Kaggle. It fetches the images and corresponding labels for training and testing purposes.
2. Scaling Images

In scaling_images.py, the images are preprocessed and scaled to a uniform size to ensure consistency across the dataset.
3. Data Augmentation

Data augmentation is performed using the data_augmentation.py script. It helps in increasing the diversity of the training dataset by applying transformations such as rotation, flipping, and zooming to the images.
4. Model Building

The architecture of the CNN model is defined in model_building.py. TensorFlow is utilized for constructing the layers of the neural network.
5. Model Training

The model_training.py script is responsible for training the constructed model on the training dataset. During training, the model learns to classify images into cat or dog categories based on the provided dataset.
6. Performance Analysis

performance_analysis.py provides insights into the performance of the trained model. It may include visualizations of training/validation accuracy and loss over epochs.
7. Model Evaluation

Finally, the model_evaluation.py script evaluates the trained model on the testing dataset to assess its accuracy and performance metrics.
Usage

    Ensure all dependencies are installed using pip install -r requirements.txt.
    Execute each script in sequential order as outlined above.
