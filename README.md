# ECG Anomaly Detection and Classification

The aim of this project is to perform **anomaly detection** and **classification** in time series using Deep Learning models, specifically focusing on health information associated with ECG signals, which record the electrical activity of the heart. To perform these tasks we use the `ECG5000` dataset, which is publicly available in the UCR Time Series Classification archive and contains a set of 5000 univariate time series, each consisting of 140 timesteps of real patient ECG data.

After analyzing the dataset and the five possible signal classes (`Normal`, `R on T`, `PVC`, `SP`, `UB`), data was pre-processed to be applied to Deep Learning models for anomaly detection. The `Keras` library was employed to implement the following Autoencoder models, training them to perform **signal reconstruction** and be able to distinguish between **normal signals** and **anomalies**, when an arrhythmia is detected:
- Autoencoder
- LSTM Autoencoder

The Autoencoders were trained only on the normal ECG sequences, in order to reconstruct these examples with minimal error: anomalous signals should have a higher reconstruction error compared to normal signals, allowing to classify a signal as an anomaly if the reconstruction error is higher than a given **threshold**. To achieve the most accurate input reconstruction, we minimized a **reconstruction loss**, specifically opting for the **Mean Squared Error (MSE)**. After computing the threshold, models were evaluated using confusion matrix, ROC-AUC curve, accuracy, precision, recall and F1-score.


In the context of the classification task, we incorporated an additional pre-processing step to address class imbalance: the **Synthetic Minority Oversampling Technique (SMOTE)**. SMOTE rebalances the class distribution within the dataset by creating new instances of minority classes (oversampling) and selectively removing instances from the majority class (undersampling). Different models were trained and evaluated, using the `Keras` library: 
- Conv1D
- LSTM
- Conv1D + LSTM

The `.ipynb` notebooks were executed using the [Google Colab](https://colab.research.google.com) platform.
