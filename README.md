# ModCloth Fit Analysis — Data Science Project

## Overview

This project analyzes the **ModCloth customer reviews dataset** to understand patterns of **fit mismatch** in apparel e‑commerce.

Fit mismatch occurs when customers report that a garment fits **too small** or **too large** instead of fitting correctly.

Understanding when and why mismatch occurs can help fashion retailers:

- reduce product returns
- improve sizing recommendations
- increase customer satisfaction
- improve product catalog decisions

The project combines **exploratory data analysis, feature engineering and user segmentation** to translate customer review data into actionable insights.

---

# Dataset

The dataset used in this project contains historical customer reviews from the ModCloth fashion retailer.

Each record includes information such as:

- product category
- purchased size
- perceived fit (small / fit / large)
- customer height
- product quality rating
- review metadata

You can download the dataset from Kaggle:

https://www.kaggle.com/code/agrawaladitya/step-by-step-data-preprocessing-eda/input?select=modcloth_final_data.json

After downloading the file, place it in:

```
data/raw/modcloth_final_data.json
```

The project also includes a **portable dataset loader** that can automatically download the dataset from a public URL of Hugging Face if the file is not present locally.

Note: the Hugging Face dataset mirror may be private and is mainly used for development convenience. The official dataset source remains Kaggle.

---

# Research Question

Can customer reviews be used to identify **systematic patterns of sizing mismatch** across categories, sizes and user profiles in order to improve the apparel e‑commerce experience?

### Analytical Hypotheses

H1 — mismatch rate varies across product categories  
H2 — mismatch rate varies across sizes  
H3 — mismatch rate varies across height buckets  
H4 — mismatch is associated with lower perceived product quality

---

# Project Structure

```
data/
    raw/
    processed/

notebooks/
    01_data_understanding.ipynb
    02_eda_strategica.ipynb

src/
    cleaning.py
    clustering.py
    data_loading.py
    metrics.py
    viz.py

reports/
    strategic_report.md
```

---

# Technologies Used

• Python

• Pandas

• NumPy

• Plotly

• Scikit-learn

• PyArrow

---

# Analytical Workflow

The analysis follows a structured pipeline:

1. Data loading and cleaning
2. Feature engineering (height parsing, mismatch flag, height buckets)
3. Exploratory data analysis (EDA)
4. Correlation analysis (Spearman)
5. Fit mismatch analysis across categories, sizes and height segments
6. User‑level feature engineering
7. User clustering
8. Translation of insights into strategic recommendations

---

# Key Insights

• Fit mismatch is **not uniformly distributed** across the product catalog.

• Certain **category–size combinations** exhibit consistently higher mismatch rates.

• Fit mismatch is associated with **lower perceived product quality ratings**.

• Customer behavior is heterogeneous: clustering reveals **distinct user segments** with different mismatch profiles.

These findings suggest that **data‑driven sizing guidance** could significantly improve the purchase experience.

---

# Strategic Implications

The analysis highlights several opportunities for fashion e‑commerce platforms:

### Data‑Driven Size Recommendations

Use historical review data to recommend sizes dynamically rather than relying on static size charts.

### Fit Reliability Indicators

Display signals such as:

```
“Customers report this item runs small”
“Most customers recommend sizing down”
```

### Product Page Improvements

For categories with high mismatch rates:

- include garment measurements
- clarify intended fit
- specify material elasticity

### Assortment Optimization

Categories with consistently high mismatch rates may require:

- improved product descriptions
- better sizing guidance
- supplier sizing standardization

---

# Reproducibility

To run the project:

1. Clone the repository
2. Download the dataset from Kaggle
3. Place the dataset in `data/raw/`
4. Run the notebooks in order:

```
notebooks/01_data_understanding.ipynb
notebooks/02_eda_strategica.ipynb
```

Required Python libraries:

```
pip install -r requirements.txt
```

---

# Limitations

The dataset lacks several variables that strongly influence garment fit:

- brand‑specific sizing standards
- garment materials and elasticity
- real garment measurements
- body shape information

Because of this, the analysis focuses on **behavioral patterns in customer reviews**, rather than physical garment characteristics.

Future work could integrate richer product metadata to improve predictive models.

---

# Future Work

Possible extensions include:

- predictive models for fit mismatch
- brand‑level analysis
- integration with return‑rate data
- experimentation with sizing recommendation systems
