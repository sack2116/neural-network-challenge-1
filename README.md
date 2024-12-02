# Student Loan Risk Prediction Using Deep Learning

## Objective
The aim of this project is to predict the likelihood of student loan repayment success using a deep neural network. By analyzing factors such as loan amount, income, and credit score, the model classifies whether a borrower is likely to repay their student loan. This analysis provides insights that could help financial institutions minimize risk and promote responsible lending practices.

---

## Questions
1. What are the most influential factors impacting student loan repayment success?
2. How can we use neural networks to predict the likelihood of repayment?
3. What insights can be derived from the model predictions to improve loan policies?

---

## Project Overview
This project uses a deep neural network built with TensorFlow's Keras API to predict loan repayment success. The data includes key borrower details, such as income, credit score, and loan amounts. The workflow involves data preprocessing, neural network modeling, and evaluating predictions to assess the model's performance.

### Technologies Used
- **Python**
- **TensorFlow / Keras**: Neural network modeling and evaluation
- **Pandas**: Data manipulation
- **Scikit-learn**: Data preprocessing and evaluation metrics

---

## Data Files
- `student-loans.csv`: The dataset containing borrower details and credit rankings.

---

## Workflow

### 1. Data Preprocessing
- **Importing Data**: Load the `student-loans.csv` file into a Pandas DataFrame.
- **Feature and Target Selection**: 
  - Features (`X`): All columns except `credit_ranking`.
  - Target (`y`): `credit_ranking`.
- **Splitting Data**: Divide the data into training and testing sets using an 80/20 split.
- **Scaling**: Use `StandardScaler` to standardize the feature data for consistency.

### 2. Neural Network Modeling
- **Architecture**:
  - Input Layer: Equal to the number of features.
  - Hidden Layers: Two layers with `relu` activation.
  - Output Layer: One neuron with `sigmoid` activation for binary classification.
- **Compilation**: The model is compiled with:
  - Loss Function: `binary_crossentropy`
  - Optimizer: `adam`
  - Metrics: `accuracy`
- **Training**: The model is trained for 50 epochs with a batch size of 32.

### 3. Model Evaluation and Prediction
- **Evaluation**: Assess the model's loss and accuracy using the test data.
- **Prediction**:
  - Make predictions on the test data.
  - Save and display results in a classification report.

### 4. Model Saving and Loading
- Save the trained model to a `Models` folder as `student_loans.keras`.
- Reload the model for prediction tasks.

### 5. Insights and Recommendations
- Generate a classification report to assess model performance.
- Discuss the potential to build a recommendation system for loan products based on the results.

---

## Output
### Key Outputs and Results:
1. **Model Summary**:
   - Displays the architecture of the deep neural network.
2. **Performance Metrics**:
   - Loss and accuracy on test data.
   - Classification report, including precision, recall, and F1-score.
3. **Model File**:
   - Saved in the `Models` folder as `student_loans.keras`.

---

## Insights and Challenges
1. **Insights**:
   - Factors such as income and credit score significantly influence loan repayment likelihood.
   - Neural networks provide robust predictions when trained with scaled data.

2. **Challenges**:
   - Handling imbalanced datasets where repayment failures are rare.
   - Ensuring interpretability of neural network predictions for stakeholders.

---

## Future Work
1. Build a recommendation system for student loans using collaborative, content-based, or context-based filtering.
2. Explore additional features or external datasets to enhance model performance.
3. Experiment with different neural network architectures to optimize accuracy and generalizability.

---

## Instructions
1. Clone this repository to your local machine.
2. Ensure all required libraries are installed.
3. Run the notebook to preprocess data, train the model, and make predictions.
4. View saved model files and results in the `Models` folder.

---

## Contact
For further information, reach out via email or the repository's issues section.
