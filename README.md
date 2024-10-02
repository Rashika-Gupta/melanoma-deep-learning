This repository contains the code and resources for predicting skin cancer (melanoma) using computer vision models. We use the dataset from the SIIM-ISIC Melanoma Classification Challenge, leveraging deep learning to classify dermoscopic images into malignant and benign categories.

🎯 Project Overview

Melanoma is the deadliest type of skin cancer, and early detection is key 🔑. In this project, we used a **Convolutional Neural Network (CNN)** to detect melanoma.

    **Model Used: SEResNext50_32x4d 
    Performance Metric: ROC-AUC 📈
    Results: Achieved a ROC-AUC of 0.987 after training for 50 epochs **🏅

📊 Feature Distributions

We explored the dataset to understand its composition. Here are some key distributions:

    Class Distribution: The dataset is highly imbalanced, with more benign samples than malignant.

    Age Distribution: How age correlates with melanoma cases.

    Anatomical Site Distribution: Location of melanoma on different parts of the body .

    Gender Distribution: Melanoma appears more frequently in males 👨 than females 👩.
![Screenshot from 2024-10-02 18-20-10](https://github.com/user-attachments/assets/0bfc7b63-6790-43ce-b84e-348dc4c8755a)

🔍 Image Augmentation

To make the model more robust, we applied various augmentation techniques (e.g., rotation, flipping, scaling). 
![Screenshot from 2024-10-02 18-13-09](https://github.com/user-attachments/assets/07151072-fb3f-4c49-b963-214d1560286e)

Below is a comparison of images before and after augmentation 🎨:
![Screenshot from 2024-10-02 18-16-38](https://github.com/user-attachments/assets/7eacfe6b-8073-4a59-984e-753c7114317d)


🧑‍💻 Model Architecture

We used SEResNext50_32x4d 🏗️, a powerful deep learning model pre-trained on ImageNet. Augmentation techniques were applied to enhance robustness.

    Batch Size: 32 
    Learning Rate: 1e-4 
    Training Time: 50 epochs 

📉 Model Training Visualizations

During the training process, we tracked key metrics like training loss, validation loss, and ROC-AUC to measure performance:

    Training Loss Curve 📉:

    Validation Loss Curve 📉:

    ROC-AUC Curve 🏅:

🏆 Results

The model was evaluated using ROC-AUC and achieved a score of 0.987, demonstrating a high ability to distinguish between malignant and benign lesions.
🔧 Future Work

Here are some steps to further enhance the project 🚀:

    Hyperparameter Tuning: Experiment with different hyperparameters to improve performance.
    Integrate Metadata: Combine patient information such as age, gender, and anatomical site into the model for more accurate predictions.
    Try New Models: Test other state-of-the-art models for better results.
