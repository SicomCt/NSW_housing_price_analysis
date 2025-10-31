# New South Wales Property Price & Type Prediction

## Project Overview
This project focuses on analyzing and predicting property prices and property types in New South Wales, Australia. The goal is to provide actionable insights for investors, buyers, and stakeholders by leveraging data-driven models.

- **Regression Task:** Predict property prices.  
- **Classification Task:** Predict property types (e.g., house, apartment, townhouse).  
- **Business Insight:** Analyze factors affecting property prices and rental yields.

---

## Dataset
- **Source:** New South Wales property sales data (`train.csv` / `test.csv`)  
- **Key Features:**
  - **Location:** `suburb`, `region`  
  - **Property attributes:** `property_size`, `num_bed`, `num_bath`, `type`  
  - **Rental information:** `median_house_rent_per_week`, `median_apartment_rent_per_week`  
  - **Suburb statistics:** `suburb_median_house_price`, `suburb_median_apartment_price`, `public_housing_pct`, population, area  
  - **Historical sales:** `date_sold`  

> Note: Sensitive or full-scale data is not included for privacy reasons. Sample or simulated datasets are used for demonstration.

---

## Data Preprocessing & Feature Engineering
- **Cleaning:**  
  - Removed irrelevant or high-missing-value columns.  
  - Handled extreme price values using IQR-based outlier replacement.  
- **Feature Engineering:**  
  - Date features: `year_sold`, `day_sold`  
  - Combined features: `num_bath_bed` = `num_bath` + `num_bed`  
  - Rental yield: `house_rent_yield`, `apartment_rent_yield`  
  - Area per room ratio: `property/room_ratio`  
  - Frequency encoding for `suburb` and `region`  
  - Binning of property size and prices  
- **Encoding:**  
  - Label Encoding for `type`  
  - Frequency Encoding for descriptive features  
- **Missing & Zero Values:**  
  - Zero values replaced with median of non-zero entries  
  - Infinite values replaced with median

---

## Modeling
### Regression: Property Price Prediction
- **Model:** XGBoost Regressor  
- **Techniques:**  
  - Log-transformation for target variable  
  - RandomizedSearchCV for hyperparameter tuning  
  - 5-fold cross-validation  
- **Evaluation:** Mean Absolute Error (MAE)

### Classification: Property Type Prediction
- **Model:** XGBoost Classifier  
- **Techniques:**  
  - Label encoding consistency across train/test  
  - Evaluation metric: `mlogloss`  
- **Evaluation:** Accuracy, Precision, Recall, F1-score

---

## Results

### Regression: Property Price Prediction
The XGBoost regression model predicts property prices with reasonable accuracy.  

- **Training MAE:** 268,270  
- **Test MAE:** 309,533  

**Insights:**  
- The model generalizes well, though test error is slightly higher, indicating some variance in unseen data.  
- Key features affecting price include property size, number of bedrooms/bathrooms, rental yield, and regional location.  

> **Visualization:**  
> *[Insert scatter plot of actual vs predicted prices]*  

---

### Classification: Property Type Prediction
The XGBoost classifier predicts property types (house, apartment, townhouse, etc.). Performance varies across property types due to class imbalance.

| Property Type | Precision | Recall | F1-score | Support |
|---------------|----------|--------|----------|--------|
| 1             | 0.85     | 0.85   | 0.85     | 86     |
| 2             | 0.75     | 0.60   | 0.67     | 5      |
| 3             | 0.00     | 0.00   | 0.00     | 2      |
| 4             | 0.00     | 0.00   | 0.00     | 8      |
| 5             | 0.94     | 0.99   | 0.97     | 866    |
| 8             | 0.00     | 0.00   | 0.00     | 1      |
| 9             | 0.50     | 0.24   | 0.32     | 21     |
| 10            | 0.00     | 0.00   | 0.00     | 0      |
| 11            | 1.00     | 0.22   | 0.36     | 9      |
| 12            | 0.53     | 0.29   | 0.37     | 35     |
| 13            | 1.00     | 0.94   | 0.97     | 16     |
| 14            | 0.42     | 0.28   | 0.33     | 18     |

**Overall Performance:**  
- **Accuracy:** 91%  
- **Macro Average:** Precision 0.50 | Recall 0.37 | F1-score 0.40  
- **Weighted Average:** Precision 0.89 | Recall 0.91 | F1-score 0.90  

**Insights:**  
- The model performs best on dominant classes (e.g., type 5), which heavily influence weighted metrics.  
- Rare property types have low recall due to limited data, highlighting a need for balanced datasets or class-specific strategies.  
- Features such as suburb, property size, and room counts are most influential for type prediction.


---

### Key Takeaways
- Rental yield and location are strong predictors of property price.  
- Model performance is excellent for common property types but limited for rare types.  
- The pipeline demonstrates end-to-end data preprocessing, feature engineering, and predictive modeling for real estate insights.  

---

## Skills Demonstrated
- Python, Pandas, NumPy  
- Feature engineering and preprocessing  
- Regression and classification modeling (XGBoost)  
- Hyperparameter tuning and cross-validation  
- Data visualization using Matplotlib and Seaborn  
- Business analysis and market insights  

---

## Usage
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/NSW_Property_Prediction.git
   cd NSW_Property_Prediction
2. Create a virtual environment and activate:
    python -m venv .venv
    .\.venv\Scripts\activate  # Windows
    source .venv/bin/activate  # macOS/Linux
3. Install required packages:
    pip install -r requirements.txt
4. Run the model scripts: