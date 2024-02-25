# classification-and-model-selection-with-kickstarter

### **Project Description:**

The project aims to classify Kickstarter projects into two categories based on their success or failure, utilizing a variety of features provided in the dataset.

### **Data Processing:**

1. **Loading Data**: The datasets **`kickstarter.gz`** and **`kickstarter_eval.gz`** were loaded into pandas DataFrames.
2. **Initial Data Inspection**: Basic inspection methods such as `.info()`,  `.head()`, `.unique()`, and `.value_counts()`were used to examine the datasets' structure, identifying feature types and potential issues like missing values.
3. **Data Cleaning**:
    - Dropping unnecessary columns that are unlikely to influence the outcome based on common sense or columns that contain too many nan values, such as 'id', 'photo', 'slug', etc. `.drop(cols, axis=1, inplace=True)`
    - Filling missing values in textual columns ('name' and 'blurb') with empty strings. `df[col].fillna(replacement, inplace=True)`
4. **Feature Engineering**:
    - Extracting categorical data from JSON-like strings in the 'category' column, focusing on general and specific categories.
        - `json.loads(x)` tidies up the json input
        - `pd.json.normalize(df.col.apply(json.loads))` can flatten the JSON strings in one column to a dataframe
    - Handling of textual data and other categorical features to prepare for model input.
        - name/blurb
            - length of string in each row `df[col].str.len()`
            - word count in each row `df[col].str.split().str.len()`
            - ave word count in each row `df['name'].apply(lambda s: np.mean([len(w) for w in s.split()]))`
        - map countries to regions such as asia, europe, etc.
    - One-hot encoding of categorical features and standardization of numerical features to ensure model compatibility.
        
        ```python
        from sklearn.preprocessing import OneHotEncoder
        
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        df_cat = pd.DataFrame(ohe.fit_transform(df_cat))
        new_column_names = ohe.get_feature_names_out(input_features=['region', 'month', 'generic_cat', 'precise_cat'])
        df_cat.columns = new_column_names
        ```
        

### **Model Selection and Training:**

- **Logistic Regression**: Initially chosen for its simplicity and interpretability. A grid search was conducted to find the optimal hyperparameters, including `solver`, `penalty`, and `C` value.
- **Challenges**: Convergence issues were encountered during the grid search, likely due to the need for more iterations and/or better feature scaling.
