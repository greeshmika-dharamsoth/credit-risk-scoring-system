# Model Performance Results

This folder contains the performance analysis visualizations generated from the Credit Risk Scoring System.

## Generating Visualizations

To generate the performance analysis images:

1. Open the original Colab notebook
2. Run all cells to train the model
3. The following images will be generated:
   - `performance_analysis.png` - 6-panel comprehensive performance visualization
   - `extended_roc_analysis.png` - 6-panel extended ROC and threshold analysis

## Expected Results

Based on the Colab notebook execution:

### Model Performance Metrics
- **Accuracy**: 87.50%
- **ROC-AUC Score**: 0.8990
- **Precision**: 89.39%
- **Recall (Sensitivity)**: 96.39%
- **F1-Score**: 92.75%

### Model Coefficients
- **Intercept**: 2.751131
- **Income**: -1.112291 (↓ Decreases Risk)
- **LoanAmount**: 1.712167 (↑ Increases Risk)
- **CreditHistory**: -0.912897 (↓ Decreases Risk)
- **WorkExperience**: 0.094141 (↑ Increases Risk)
- **HomeOwnership**: 0.248981 (↑ Increases Risk)

### Confusion Matrix
- True Negatives (TN): 15
- False Positives (FP): 19
- False Negatives (FN): 6  
- True Positives (TP): 160

### ROC-AUC Analysis
- **Optimal Threshold**: 0.8158
- **True Positive Rate at Optimal**: 0.8133
- **False Positive Rate at Optimal**: 0.1765
- **Specificity**: 0.8235
- **Discriminative Power**: 0.6368

## Visualization Contents

### performance_analysis.png
6-panel comprehensive visualization including:
1. Confusion Matrix
2. ROC-AUC Curve
3. Precision-Recall Curve
4. Model Coefficients (Feature Importance)
5. Performance Metrics Bar Chart
6. Prediction Probability Distribution

### extended_roc_analysis.png
6-panel extended analysis including:
1. Detailed ROC Curve with Optimal Threshold Marker
2. Metrics vs Classification Threshold
3. Sensitivity & Specificity vs Threshold
4. Feature Correlation with Target
5. Cumulative Gain Chart
6. Model Performance Summary Table

## Note

Run the Colab notebook to generate these visualizations. The images can then be downloaded and placed in this folder for documentation purposes.
