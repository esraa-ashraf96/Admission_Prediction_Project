# Admission Prediction Project

## Overview

This project aims to predict the admission chances of students into a university based on various factors such as GRE score, TOEFL score, university rating, statement of purpose (SOP), letter of recommendation (LOR), CGPA, and research experience. The project utilizes decision tree classification and regression models to predict admission outcomes.

## Dataset

The dataset used for this project is the "Admission_Predict.csv" file, which contains information about 400 applicants, including their GRE scores, TOEFL scores, university ratings, SOP scores, LOR scores, CGPA, research experience, and chance of admission.

## Workflow

1. **Data Preprocessing:** The dataset is loaded using pandas and preprocessed to add a new column, 'Admitted', which categorizes the 'Chance of Admit' data into binary values based on a threshold (0.72). 

2. **Exploratory Data Analysis (EDA):** EDA is performed using seaborn and matplotlib to visualize the relationships between different features and the target variable.

3. **Decision Tree Classification:** A decision tree classifier is trained on the dataset to predict the 'Admitted' column, which represents whether a student is likely to be admitted based on the given features.

4. **Model Evaluation:** The classification model is evaluated using various metrics such as accuracy, precision, recall, F1 score, ROC AUC score, confusion matrix, and classification report.

5. **Decision Tree Regression:** Another decision tree model is trained on the dataset to predict the 'Chance of Admit' column as a numerical value.

6. **Regression Model Evaluation:** The regression model is evaluated using metrics such as explained variance score, max error, mean squared error, mean squared logarithmic error, median absolute error, and R-squared score.

7. **Model Visualization:** Decision tree models are visualized using the plot_tree function from sklearn to understand the decision-making process of the models.

8. **Conclusion:** The project concludes that decision tree models are not suitable for this dataset, as they tend to overfit or underfit the data. 

## Dependencies

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `Admission_Prediction_project.ipynb` to execute the project.

## Future Improvements

- Try different machine learning algorithms such as Random Forest, Gradient Boosting, or Neural Networks.
- Perform feature engineering to create new features that might improve the model's performance.
- Collect more data to improve the model's accuracy and generalization.

## Contact Information

For any questions or feedback, please contact Esraa Ashraf Abd El-Aziz at esraa.ashraf1996@yahoo.com.
