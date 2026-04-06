# USA Real Estate — ML Data Preprocessing & Modeling

**Dataset:** [USA Real Estate Dataset (Kaggle)](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)  
**Target variable:** `price`  
**Dataset size:** 2,226,382 listings across the United States

---

## Dataset Overview

The dataset contains real estate listings scraped from Realtor.com with the following columns:

| Column | Type | Notes |
|--------|------|-------|
| `price` | Continuous | Target variable — listing/sale price in USD |
| `bed` | Continuous | Number of bedrooms |
| `bath` | Continuous | Number of bathrooms |
| `acre_lot` | Continuous | Lot size in acres |
| `house_size` | Continuous | Interior size in sq ft |
| `status` | Categorical | for_sale / sold / ready_to_build |
| `state` | Categorical | 55 unique values |
| `city` | Categorical | 20,098 unique values |
| `zip_code` | Categorical | 30,334 unique values (stored as float, treated as ID) |
| `brokered_by` | Categorical | Broker ID — dropped (no signal) |
| `street` | Categorical | Street ID — dropped (no signal) |
| `prev_sold_date` | Categorical | 700k+ missing — dropped |

---

## Feature Analysis

We split the features into continuous and categorical groups. The continuous ones — `bed`, `bath`, `acre_lot`, and `house_size` — are all numeric measurements with a natural range. `bed` and `bath` are integers but behave like continuous variables given their ordering.

The categorical group includes `status`, `state`, `city`, `zip_code`, `brokered_by`, `street`, and `prev_sold_date`. Some of these are stored as numbers in the file (like `brokered_by` and `zip_code`) but they're really just ID codes with no meaningful numeric ordering.

For usefulness in predicting price, `house_size` stands out as probably the most informative continuous feature — larger homes cost more. `bed` and `bath` are correlated with house size so they add some signal but not as much independently. `acre_lot` is more useful in rural areas and noisier in urban ones. Location features like `state` and `city` are probably the strongest predictors overall — the median price varies enormously across states, with northeast states like New York and New Jersey sitting much higher than the south and midwest. `zip_code` captures even finer location signal but is harder to encode.

We created one engineered feature — `bath_per_bed` (bath-to-bedroom ratio) — which captures how well-appointed a home is relative to its size. We also computed a scatter matrix to visualize pairwise relationships between all continuous features and price, and a Pearson correlation bar chart.

`brokered_by` and `street` are essentially arbitrary ID numbers so we dropped them. `prev_sold_date` has over 700k missing values and would require significant feature engineering to be useful, so it was dropped as well.

---

## Distribution Analysis & Data Issues

The price distribution is extremely right-skewed. Most listings are under $1M but there are values going up to $2 billion which are clearly bad data entries — the mean ($524k) is far above the median ($325k). A log transform on price would help significantly before modeling.

The continuous predictors have the same problem. `house_size` has a maximum value of around 1 billion sq ft which is physically impossible — most homes are between 1,300 and 2,400 sq ft. `bed` has a maximum of 473 and `bath` has similarly extreme outliers. `acre_lot` is heavily right-skewed too. These are all data entry errors that need to be capped or removed before training a model on this data.

On the categorical side, listings are dominated by "for_sale" status. Looking at price by status, sold prices tend to run a bit lower than listed prices which makes intuitive sense. State shows strong variation in median price — the location signal is real and worth capturing.

Before serious modeling, the data would benefit from:
- Capping or removing extreme outliers in price, house_size, bed, bath, and acre_lot
- A log transform on price (and possibly acre_lot and house_size) to reduce skew
- Proper encoding of categorical columns
- Handling the missing values spread across multiple columns

---

## Train / Test Split

We used an 80/20 split — 80% for training and 20% for testing. With 2.2 million rows the test set is still around 440k records, more than enough for a reliable evaluation. We set `random_state=42` to keep the split reproducible across runs.

We chose not to use stratified sampling. Stratification is most useful for classification problems where you need each class proportionally represented. Here price is a continuous regression target so you can't stratify on it directly. We could have stratified on `status`, but with this many rows the random shuffle already produces nearly identical proportions in train and test — we confirmed this by comparing the status distributions in both splits.

---

## Handling Missing Values

The features with meaningful missing data are `house_size` (~25% missing), `bath` (~23%), `bed` (~22%), and `acre_lot` (~15%). `city` and `zip_code` have under 1% missing.

We chose to impute rather than drop rows. Dropping would mean losing around 450k training rows — about a quarter of the data — which is too much to throw away, especially since the missing values are spread across many different records rather than concentrated in a few bad ones.

For the numeric features we used **median imputation**. These columns all have heavy right skew and extreme outliers, so the mean gets pulled way up (a 473-bedroom listing would inflate the mean for `bed` significantly). The median is a much more sensible fill value in this situation.

For the categorical columns with missing values we used **most frequent imputation**. With under 1% missing, it barely matters what strategy we use — most frequent is simple and keeps those rows usable.

We were careful to fit the imputers only on the training data and then apply them to the test set. Fitting on the full dataset would let test information leak into training, which inflates evaluation metrics and gives a false picture of model performance.

---

## Preprocessing Pipeline

We built a full scikit-learn `Pipeline` for each model so all preprocessing steps — imputation, encoding, and scaling — happen in one shot with no risk of data leakage.

The preprocessor uses `ColumnTransformer` to apply different transformations to different columns:

- **Numeric features** (`bed`, `bath`, `acre_lot`, `house_size`, `bath_per_bed`): median imputation followed by `StandardScaler`. Scaling matters especially for Linear Regression — without it, `house_size` in the thousands would dominate over `bath` which is usually 1–4.

- **Low-cardinality categoricals** (`status`, `state`): most frequent imputation followed by `OneHotEncoder`. With only 3 and 55 unique values respectively, one-hot encoding is manageable. We first demonstrated `OrdinalEncoder` to show why it's problematic here — assigning integers implies a false ordering (e.g. `for_sale=0 < sold=2`) that doesn't exist in the data.

- **High-cardinality categoricals** (`city`, `zip_code`): most frequent imputation followed by `TargetEncoder`. One-hot encoding `city` (20k values) and `zip_code` (30k values) would create 50k+ columns across 2.2M rows — that blows up memory. `TargetEncoder` instead replaces each category with the mean price for that category from the training data, keeping it as a single column while still capturing the location signal.

---

## Model Results

We trained Linear Regression and Decision Tree Regressor with default parameters and evaluated on the test set.

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | — | — | 0.18 |
| Decision Tree | — | — | 0.11 |

Linear Regression edged out the Decision Tree despite the non-linear nature of the data. The reason is that with default parameters the Decision Tree grows extremely deep on 2.2M rows, memorizing the training data but failing to generalize — classic overfitting. Linear Regression fits a stable global pattern even if it misses the non-linear effects.

Both R² scores are low overall, which isn't surprising given the state of the raw data. The extreme price outliers (up to $2 billion) completely throw off both models — a handful of bad records contribute massively to the error. With outlier capping and a log transform on price, both models would perform significantly better. With default parameters on unprocessed data, these numbers are about what you'd expect.

---

## Feature Scaling Comparison

We compared three scaling methods on `house_size` and `bath_per_bed` to illustrate how each handles the data differently:

**StandardScaler** subtracts the mean and divides by standard deviation, centering features at 0 with unit variance. The shape of the distribution doesn't change — skew and outliers are still there, just rescaled. This is the most common default choice and works well when features are roughly normally distributed.

**MinMaxScaler** compresses everything into [0, 1]. The problem with this dataset is that one extreme outlier in `house_size` drags all the normal values down near zero — the entire typical range of houses gets compressed into a tiny slice. The histograms show this clearly with almost all values piling up near zero.

**RobustScaler** uses the median and IQR instead of mean and standard deviation, making it much less sensitive to outliers. For `house_size` this makes a significant difference — the bulk of the distribution spreads out naturally even with extreme values present. For messy real-world datasets like this one, RobustScaler is generally the better choice.

We used StandardScaler in the pipeline as a reasonable baseline. Switching to RobustScaler would likely improve model performance given the outlier situation in this dataset.

---

## Setup

```bash
# create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# download the dataset (requires Kaggle API credentials)
kaggle datasets download -d ahmedshahriarsakib/usa-real-estate-dataset --unzip

# run notebook
jupyter notebook
```

> The dataset (`realtor-data.zip.csv`) is not included in this repo as it exceeds GitHub's 100MB file size limit.
> Download it directly from [Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) and place it in the root of the project folder.
