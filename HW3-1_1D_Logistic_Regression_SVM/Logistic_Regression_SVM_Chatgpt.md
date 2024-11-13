## Prompt:
Implement a system to perform Logistic regression and Support Vector Machine (SVM) on a set of 300 randomly generated variables, and visualize the results.

1. Generate Data:
   Generate 300 random variables, X(i), with values ranging from 0 to 1000.
   Determine Y(i) based on the conditio: Y(i) = 1 if 500 < X(i) < 800, otherwise Y(i) = 0.

2. Model Building:
   Implement a Logistic Regression model to predict the binary outcomes, outputting results as (y1).
   Implement a Support Vector Machine (SVM) model to predict the binary outcomes, outputting results as (y2).

3. Visualization:
   Create a separate plot for actual data (X and Y) alongside Logistic Regression predictions (X, Y1).
   Create a separate plot for actual data (X and Y) alongside Support Vector Machine (SVM) predictions (X, Y2).

Notes:
- Give me complete code , Following crisp-dm steps.
- Utilize 'numpy' libraries for random, 'scikit-learn' libraries for machine learning and 'matplotlib' for visualization in Python.
- Randomness in data generation should be controlled (e.g. using a fixed seed) to allow reproducibility.
- For SVM, use rbf kernel to process this non-linear problem.
## Response:
Here's the complete code, following the CRISP-DM steps, to implement a Logistic Regression and SVM model for predicting binary outcomes based on random data. This includes data generation, model training, predictions, and visualizations.

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Step 1: Business Understanding
# Goal: Predict binary outcomes based on random data using Logistic Regression and SVM.

# Step 2: Data Understanding
# Generate 300 random variables with values between 0 and 1000. 
# Set target variable Y based on the rule: Y(i) = 1 if 500 < X(i) < 800, otherwise Y(i) = 0.

# Set random seed for reproducibility
np.random.seed(42)

# Generate 300 random X values between 0 and 1000
X = np.random.randint(0, 1000, 300).reshape(-1, 1)

# Generate Y based on the given condition
Y = np.where((X > 500) & (X < 800), 1, 0).ravel()

# Step 3: Data Preparation
# No further preparation is needed since data is already in numerical format and follows the conditions.

# Step 4: Modeling

# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X, Y)
y1 = logistic_model.predict(X)

# Support Vector Machine model with rbf kernel
svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X, Y)
y2 = svm_model.predict(X)

# Step 5: Evaluation
# Predictions (y1 for Logistic Regression and y2 for SVM) will be evaluated visually alongside actual data.

# Step 6: Deployment
# Visualization of results using matplotlib.

# Visualization for actual data (X, Y) and Logistic Regression predictions (X, Y1)
plt.figure(figsize=(12, 6))

# Plot actual data
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.scatter(X, y1, color='orange', alpha=0.5, label='Logistic Regression Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logistic Regression vs Actual')
plt.legend()

# Plot actual data and SVM predictions
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.scatter(X, y2, color='green', alpha=0.5, label='SVM Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM vs Actual')
plt.legend()

plt.tight_layout()
plt.show()
```

### Explanation of Each Step

1. **Data Generation**:

   * We generate 300 random integer values for <math><semantics><mrow><mi>X</mi></mrow><annotation>X</annotation></semantics></math>X between 0 and 1000.
   * We set <math><semantics><mrow><mi>Y</mi><mo>\=</mo><mn>1</mn></mrow><annotation>Y = 1</annotation></semantics></math>Y\=1 if <math><semantics><mrow><mn>500</mn><mo><</mo><mi>X</mi><mo><</mo><mn>800</mn></mrow><annotation>500 < X < 800</annotation></semantics></math>500<X<800; otherwise, <math><semantics><mrow><mi>Y</mi><mo>\=</mo><mn>0</mn></mrow><annotation>Y = 0</annotation></semantics></math>Y\=0.

2. **Modeling**:

   * A Logistic Regression model is created and trained on the data to predict <math><semantics><mrow><mi>Y</mi></mrow><annotation>Y</annotation></semantics></math>Y.
   * An SVM model with an RBF kernel is also trained to capture the non-linear relationship between <math><semantics><mrow><mi>X</mi></mrow><annotation>X</annotation></semantics></math>X and <math><semantics><mrow><mi>Y</mi></mrow><annotation>Y</annotation></semantics></math>Y.

3. **Visualization**:

   * Two plots are created:
     * One showing the actual <math><semantics><mrow><mi>Y</mi></mrow><annotation>Y</annotation></semantics></math>Y values alongside the Logistic Regression predictions.
     * Another showing the actual <math><semantics><mrow><mi>Y</mi></mrow><annotation>Y</annotation></semantics></math>Y values alongside the SVM predictions.
   * This allows us to visually assess the performance of each model.

## Prompt:
4\. Visualization: Create a plot for actual data (X and Y) and Logistic Regression predictions, including relevant decision boundaries (hyperplanes) in respective plots. Create a plot for actual data (X and Y) and Support Vector Machine (SVM) predictions, including relevant decision boundaries (hyperplanes) in respective plots. # Notes - Ensure that plots clearly depict the data points and decision boundaries for easy interpretation. # Output Format : (Two plots should be generated) 1. A plot with actual data (X, Y) and Logistic Regression results (X, Y1), including the decision boundary. 2. A plot with actual data (X, Y) and Support Vector Machine (SVM) results (X, Y2), including the decision boundary.

## Response:
Here's the updated code to include decision boundaries in the plots for both Logistic Regression and SVM models.

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Set random seed for reproducibility
np.random.seed(42)

# Generate 300 random X values between 0 and 1000
X = np.random.randint(0, 1000, 300).reshape(-1, 1)

# Generate Y based on the given condition
Y = np.where((X > 500) & (X < 800), 1, 0).ravel()

# Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X, Y)
y1 = logistic_model.predict(X)

# Support Vector Machine model with rbf kernel
svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X, Y)
y2 = svm_model.predict(X)

# Create a range for decision boundary plotting
X_plot = np.linspace(0, 1000, 500).reshape(-1, 1)

# Logistic Regression decision boundary
logistic_boundary = logistic_model.predict_proba(X_plot)[:, 1] > 0.5

# SVM decision boundary
svm_boundary = svm_model.decision_function(X_plot)

# Plot for Logistic Regression with decision boundary
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.scatter(X, y1, color='orange', alpha=0.5, label='Logistic Regression Prediction')
plt.plot(X_plot, logistic_boundary, color='red', linewidth=2, label='Decision Boundary')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Logistic Regression vs Actual')
plt.legend()

# Plot for SVM with decision boundary
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.scatter(X, y2, color='green', alpha=0.5, label='SVM Prediction')
plt.plot(X_plot, (svm_boundary > 0).astype(int), color='purple', linewidth=2, label='Decision Boundary')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVM vs Actual')
plt.legend()

plt.tight_layout()
plt.show()
```

### Explanation:

* **Logistic Regression**: The decision boundary is plotted at the probability threshold (0.5).
* **SVM**: The decision boundary is defined using the sign of the decision function output, marking the classification threshold.

These plots visually represent the classification accuracy and decision boundaries for each model.

