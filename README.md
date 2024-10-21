# Heart Failure Survival Prediction

This project uses a machine learning model to predict the survival of patients with heart failure based on key factors such as serum creatinine, ejection fraction, age, anemia, diabetes, and other health indicators. The model aims to support early detection and management of cardiovascular disease (CVD), the leading cause of death globally.

## Project Overview

Cardiovascular diseases (CVDs) are responsible for an estimated 17.9 million deaths each year, accounting for 31% of all deaths worldwide. Heart failure is often a result of CVDs, and early detection and effective management of risk factors are crucial in reducing mortality. This project utilizes a Kaggle dataset of heart failure patients with 12 key features to build a classification model that predicts the likelihood of survival.

This project is primarily a hands-on exercise for me to get familiar with using Git, exploring machine learning techniques, and working with libraries like TensorFlow and Scikit-learn. It was a learning experience inspired by resources from Codecademy and guidance from ChatGPT.

### Dataset

The dataset contains the following features:
- `age`: Age of the patient
- `anaemia`: Presence or absence of anemia
- `creatinine_phosphokinase`: Level of CPK enzyme in blood (mcg/L)
- `diabetes`: Presence or absence of diabetes
- `ejection_fraction`: Percentage of blood leaving the heart with each contraction
- `high_blood_pressure`: Whether the patient has hypertension
- `platelets`: Platelet count in blood (kiloplatelets/mL)
- `serum_creatinine`: Level of creatinine in blood (mg/dL)
- `serum_sodium`: Level of sodium in blood (mEq/L)
- `sex`: Gender of the patient (Male/Female)
- `smoking`: Whether the patient is a smoker
- `time`: Follow-up period (days)
- `DEATH_EVENT`: Whether the patient died during the follow-up period

### Goals

- Predict patient mortality due to heart failure using machine learning techniques.
- Utilize early detection strategies to highlight risk factors and support interventions.
- Build a classification model using key features that contribute to cardiovascular disease risk.

## Project Workflow

1. **Data Preprocessing**:
    - Handled missing values (if any).
    - Scaled numeric features using `StandardScaler`.
    - Encoded categorical labels with `LabelEncoder` and transformed them into binary vectors.

2. **Model Design**:
    - Built a neural network model using TensorFlow and Keras.
    - The model consists of an input layer, one hidden layer with 12 neurons, and an output layer using softmax activation for classification.
    - Compiled the model with categorical cross-entropy loss, Adam optimizer, and accuracy metrics.

3. **Training**:
    - Trained the model on the dataset with 100 epochs and a batch size of 16.
    - Used 80% of the data for training and 20% for validation.

4. **Evaluation**:
    - Evaluated the model performance using accuracy, confusion matrix, and classification report.
    - Visualized training history for accuracy and loss.

5. **Model Metrics**:
    - Confusion Matrix to evaluate model performance.
    - Classification Report with precision, recall, and F1-scores.

## Key Visualizations

- **Training History**: Visualized the model's training and validation loss and accuracy over epochs.
- **Confusion Matrix**: Illustrated true vs. predicted classifications.
- **Class Distribution**: Displayed the distribution of the classes (survived vs. not survived).


## Future Work

- **Feature Importance**: Investigate feature importance using techniques like SHAP or permutation importance to understand the contribution of each feature.
- **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to optimize hyperparameters for better performance.
- **Imbalanced Classes**: Explore techniques such as oversampling or undersampling if class imbalance significantly affects model performance.

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/heart-failure-prediction.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the script:
    ```bash
    python main.py
    ```

## Requirements

- TensorFlow
- Scikit-learn
- Matplotlib
- Numpy
- Pandas

## References

- [Kaggle Heart Failure Dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



