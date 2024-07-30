# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# **Acceptance Study for Learning Analytics: Reduce/impute the data**

# ## Load packages and data

import pandas as pd
import numpy as np
# from IPython.display import display # not necessary in Jupyter
# import os

# +
# Load data from excel file, uncomment these 2 commands:
# acceptance_df = pd.read_excel('./data/data_merged.xlsx')
# # Set "SurveyID" as index column
# acceptance_df.set_index("SurveyID", inplace = True)

# To load the data from csv, uncomment this line:
acceptance_df = pd.read_csv("./data/clean_data/data_merged.csv",index_col="SurveyID")
acceptance_df.head(3)
# -

# ## Not promising (superfluous) columns, consider removing later

# +
# superfluous_columns = [
#     "1.2", "1.3", "1.4", "2.2", "2.4", "2.6", "2.10", "2.15", "2.16", "2.17",
#     "2.19", "2.21", "2.22", "2.23", "2.24"
# ]
# acceptance_df = acceptance_df.drop(columns = superfluous_columns)
# -

# ## Missing data

# +
# Drop or impute values
# Copying data
acceptance_reduced_df = acceptance_df.copy()
acceptance_imputed_df = acceptance_df.copy()

# Drop values
acceptance_reduced_df.dropna(inplace=True)
# Reconstruct the index to avoid jumps
acceptance_reduced_df.index = pd.RangeIndex(
    start=1, stop=len(acceptance_reduced_df)+1,
    name="SurveyID")
# -

# Prior to imputation, we check the NA shares row-wise. We may then drop all the rows with many NAs and impute the rest afterwards:

# +
na_observe_df = acceptance_df.copy()
na_observe_df["na_share"] = acceptance_df.isnull().sum(axis=1)/acceptance_df.shape[1]
na_observe_df = na_observe_df.iloc[:,-1:]
na_observe_df = round(100*na_observe_df)

for i in np.arange(10,100,step=10):
    print(
        len(na_observe_df[na_observe_df["na_share"]>i]),
        f"rows contain more than {i} % NAs"
    )
# -

# Let us have a look on the rows with more than 45% NAs:

na_descending_df = na_observe_df.sort_values(by = "na_share", ascending=False)
na_descending_df = na_descending_df[na_descending_df["na_share"]>45] # rows with >50% NAs
na_descending_df

# The imputation of the above rows makes no sense, that would just increase the variance in the data set. It is much more safe to drop these rows. Thus, we consider 45% as a <font color="crimson">NA threshold</font> and we'll drop 9 rows out of 562:

acceptance_imputed_df = acceptance_imputed_df[na_observe_df["na_share"] <= 45]
print("Number of remained rows:", len(acceptance_imputed_df))
acceptance_imputed_df.iloc[180:182,:] # check that row no. 181 was removed

# Reconstruct the index to avoid jumps/gaps
acceptance_imputed_df.index = pd.RangeIndex(
    start=1, stop=len(acceptance_imputed_df)+1,
    name="SurveyID")
acceptance_imputed_df.iloc[180:182,:] # check the previous rows with new indices

# Before imputing, we will export this optimized data set for further references:

acceptance_imputed_df.to_csv('./data/clean_data/data_optimized.csv',index_label="SurveyID")

# Anytime, if we want to deal with unimputed data, we will use `data_optimized.csv`, because it is obtained from the raw data by dropping only 6+9 NA rows.
#
# Now we will proceed to imputation:

# +
# Impute values
# Method 1 - pandas build-in imputing method
# Impute the first column (specializations) with most frequent values (mode) / median
acceptance_imputed_df["1.1"] = acceptance_imputed_df["1.1"].fillna(acceptance_imputed_df["1.1"].mode().iloc[0])
# acceptance_imputed_df["1.1"] = acceptance_imputed_df["1.1"].fillna(acceptance_imputed_df["1.1"].median())
# Alternative approach: Consider adding a new specialization value "unknown" instead

#  Impute the rest with mean
acceptance_imputed_df = acceptance_imputed_df.fillna(acceptance_imputed_df.mean())
# Note: Consider the implementation of a NA Threshhold. E.g., we can
# delete all the rows containing more than 5 NAs and impute the rest

# Method 2 - sklearn imputer
# from sklearn.impute import SimpleImputer # or KNNImputer (smarter)
# imputer = SimpleImputer(strategy="mean")
# imputer.fit(acceptance_imputed_df[['2.1']]) # call the "fit" method on the column
# acceptance_imputed_df['2.1'] = imputer.transform(acceptance_imputed_df[['2.1']]) # fill the column

# Convert the column "1.1" (specializations) to integer
acceptance_reduced_df["1.1"] = acceptance_reduced_df["1.1"].astype('int')
acceptance_imputed_df["1.1"] = acceptance_imputed_df["1.1"].astype('int')
# Uncomment the following commands to convert all the columns to integer
# acceptance_reduced_df = round(acceptance_reduced_df).astype('int')
# acceptance_imputed_df = round(acceptance_imputed_df).astype('int')
# -

# ## Comparison / check-up

# +
# View sizes of reduced and imputed data:
data_sizes_df = pd.DataFrame(
    {
        "data": ["reduced","imputed"],
        "observations": [acceptance_reduced_df.shape[0], acceptance_imputed_df.shape[0]]
    },
)

data_sizes_df = data_sizes_df.set_index("data")

#  Check data sizes:
data_sizes_df
# -

# We observe that we  would lose 214 rows if we drop all NAs!
#
# To view the data comfortably, we adjust pandas displaying options:

# +
# View the default values
# pd_max_rows = pd.get_option("display.max_rows") # maximum number of displayed rows / 60 or 10
# pd_max_cols = pd.get_option("display.max_columns") # maximum number of displayed columns / 20
# pd_precision = pd.get_option('display.precision') # float display precision / 6
# pd_float_format = pd.options.display.float_format # float display format / None
# pd_max_rows, pd_max_cols, pd_precision, pd_float_format # View the values

# Customize pandas displaying options
# pd.set_option("display.max_rows",acceptance_imputed_df.shape[0]+1) # to view all rows
# pd.set_option("display.max_columns",acceptance_imputed_df.shape[1]+1) # to view all columns
# pd.set_option('display.precision',2) # display 2 decimal places

# Restore pandas defaults
# pd.set_option("display.max_rows",pd_max_rows) # restore defaults
# -

# Set pandas display options temporarily to view "acceptance_imputed_df":
with pd.option_context("display.max_rows", acceptance_reduced_df.shape[0]+1,
                       "display.max_columns", acceptance_reduced_df.shape[1]+1,
                       'display.precision',2):
    display(acceptance_reduced_df)

# Set pandas display options temporarily to view "acceptance_imputed_df":
with pd.option_context("display.max_rows", acceptance_imputed_df.shape[0]+1,
                       "display.max_columns", acceptance_imputed_df.shape[1]+1,
                       'display.precision',2):
    display(acceptance_imputed_df)

# ## Export data

acceptance_reduced_df.to_excel('./data/clean_data/data_reduced.xlsx')
acceptance_imputed_df.to_excel('./data/clean_data/data_imputed.xlsx')
acceptance_imputed_df.to_csv('./data/clean_data/data_imputed.csv',index_label="SurveyID")
data_sizes_df.to_csv('./data/clean_data/data_sizes.csv',index_label="data")
