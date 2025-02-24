# sales-prediction
Machine learning project for predicting sales based on advertising factors.
# Sales Prediction Project

This project focuses on predicting sales based on advertising expenditures in various media channels such as TV, Radio, and Newspaper. The goal is to build a machine learning model to forecast future sales using these advertising budgets.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Steps Involved](#steps-involved)
5. [Model Evaluation](#model-evaluation)


## Project Overview

The main objective of this project is to predict sales using a dataset that includes information about advertising budgets across TV, Radio, and Newspaper. I used a **Linear Regression** model to analyze the relationship between the advertising expenditures and sales, and I also evaluated the model's performance.

## Dataset

The dataset used for this project is `Advertising.csv`, which contains the following columns:

- `TV`: Advertising budget spent on TV.
- `Radio`: Advertising budget spent on Radio.
- `Newspaper`: Advertising budget spent on Newspaper.
- `Sales`: The actual sales generated by the product.

### Sample of the Dataset

| TV    | Radio | Newspaper | Sales |
|-------|-------|-----------|-------|
| 230.1 | 37.8  | 69.2      | 22.1  |
| 44.5  | 39.3  | 45.1      | 10.4  |
| 17.2  | 45.9  | 69.3      | 9.3   |
| 151.5 | 41.3  | 58.5      | 18.5  |

## Technologies Used

This project uses the following technologies:

- **Python**: The main programming language for this project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Matplotlib/Seaborn**: For data visualization.
- **Scikit-learn**: To build and evaluate the machine learning model.
- **Joblib**: To save and load the trained model.

## Steps Involved

1. **Load the Dataset**:
   - The data is loaded from the `Advertising.csv` file using **Pandas**.
   
2. **Explore the Dataset**:
   - The dataset is explored by displaying the first few rows and summarizing the data (statistics, missing values, etc.).

3. **Handle Missing Values**:
   - Any missing data is removed from the dataset to ensure model reliability.

4. **Data Visualization**:
   - A correlation heatmap is created to understand how the features relate to each other and to the target variable (Sales).

5. **Feature Selection and Preprocessing**:
   - Relevant features (`TV`, `Radio`, `Newspaper`) are selected, and the features are scaled using **StandardScaler** to normalize the data.

6. **Train a Linear Regression Model**:
   - The dataset is split into training and testing sets, and a **Linear Regression** model is trained on the training data.

7. **Model Evaluation**:
   - The model's performance is evaluated using **Mean Squared Error (MSE)** and **R-squared** scores.
   - Cross-validation is performed to ensure the model's stability.

8. **Save the Model**:
   - The trained model is saved using **Joblib** so it can be reused later.

9. **Visualize Predictions**:
   - **Actual vs Predicted Sales** are visualized to compare the model’s performance.
   - Feature importance is also visualized to show how each advertising channel impacts sales.

## Model Evaluation

After training the model, its performance is assessed using:

- **Mean Squared Error (MSE)**: A measure of the average squared difference between actual and predicted values.
- **R-squared**: The proportion of variance in the target variable explained by the features.
- **Cross-Validation**: Evaluating the model's performance over multiple subsets of the data to ensure its reliability.

