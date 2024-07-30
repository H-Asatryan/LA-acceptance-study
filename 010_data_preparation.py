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

# # Acceptance Study: Data preparation and the first report

# In this piece of code, we import the data, fix the encoding and replace the long column descriptions by abbreviations / question codes, export the question abbreviations and the data for further processing, merge data chunks and export the information about chunk sizes for later cases. Afterwards we remove the rows consisting of all NaNs, as well as we create the first report on the data using Pandas Profiling (ydata-profiling).

# ## Load packages

# ### Install packages on demand and load them

# +
# Uncomment the following lines to install "ydata_profiling" on demand
# try:
#     import ydata_profiling
# except:
# #     !pip install ydata_profiling
# -

import pandas as pd
import ydata_profiling # import pandas_profiling
import os
from functions_and_variables import check_and_create_directory


# ### Create the directory structure

# +
# Build the directory structure
def check_and_create_directory(directory_name):
  """Checks if the directory exists and creates it if it doesn't."""
  if not os.path.exists(directory_name):
    os.mkdir(directory_name)

# # !mkdir output
# # !pwd
directory_list = [
    "output",
    "output/clustering",
    "output/clustering/dbscan",
    "output/clustering/kmeans",
    "output/clustering/manual",
    "output/clustering/spectral",
    "output/clustering/ward",
    "output/clustering/kmeans/direct_with_pca",
    "output/clustering/kmeans/direct_without_pca",
    "output/clustering/kmeans/via_row_avg_ft",
    "output/docs",
    "output/docs/dataset_info",
    "output/docs/reports",
    "output/extra",
    "output/extra/hypotheses"
]

for folder in directory_list:
    check_and_create_directory(folder)
# -

# ## Load data chunks and export the question names

# We first load the raw data and create short column names for it:

# +
# Read the 1st data chunk and fix its encoding:
acceptance_raw_df_1 = pd.read_csv("./data/raw_data/raw_data.csv",
                                  sep=";",
                                  encoding='cp1252')
# Another fixing approach, which removes the first row automatically
# Not desired if we need to save questions
# acceptance_raw_df_1 = pd.read_csv('./data/raw_data.csv', encoding='1252', delimiter=';', skiprows=1)

# Read the 2nd data chunk and fix its encoding:
acceptance_raw_df_2 = pd.read_csv("./data/raw_data/raw_data_new_SS2023.csv",
                                  sep=";",
                                  encoding='cp1252',
                                  index_col="Bogen")

acceptance_df_1 = acceptance_raw_df_1.copy()
acceptance_df_2 = acceptance_raw_df_2.copy()

# Select only first 1+4+24 columns, i.e., drop the last columns containing student opinions
acceptance_df_1 = acceptance_df_1.iloc[:, :29]
acceptance_df_2 = acceptance_df_2.iloc[:, :29]

# Retrieving column names from the first chunk
# Note: The raw structures of both chunks are different!
# Questions of the first section:
questions_1 = list(acceptance_df_1.iloc[0, 1:5])

# The 2nd part of the sheet, which is called "Learning Analytics" has 24 coupled questions.
# We retreive these 12 common titles:
common_titles = acceptance_df_1.columns.values[5:29:2]
common_titles = [title[3:] for title in common_titles]
# Each of these common titles has two possible subtitles:
common_subtitles = list(acceptance_df_1.iloc[0, 5:7])
questions_2 = []
for common_title in common_titles:
    questions_2.append(common_title + " -> " + common_subtitles[0])
    questions_2.append(common_title + " -> " + common_subtitles[1])

questions = questions_1 + questions_2

# Generate shorter column names
short_col_names = list("1."+str(num) for num in range(1,5))+\
list("2."+str(num) for num in range(1,25))

questions_df = pd.DataFrame({
    "question_code": short_col_names,
    "question": questions
})
questions_df.set_index('question_code', inplace=True)

# Write DataFrame to Excel file
questions_df.to_excel('./output/docs/question_abbreviations.xlsx')
# questions_df.head()
# -

# ## Shorten column names and merge chunks

# Now we will check duplicated rows:

# Uncomment to check the number of duplicated rows
acceptance_df_1.duplicated().sum()+acceptance_df_2.duplicated().sum()
# We have no duplicates, fine!

#  Next, we shorten the column names:

# +
# Chunk 1: Dealing with the strange two-line header
# Drop the first row / duplicated column names
# acceptance_df_1 = acceptance_df_1.tail(-1)
acceptance_df_1 = acceptance_df_1.iloc[1:]
# Shorten the column names
acceptance_df_1.columns = ["SurveyID"]+short_col_names
# Set "SurveyID" as an index
acceptance_df_1.set_index("SurveyID", inplace = True)

# Chunk 2: Drop the superfluous column "Seriendruck-ID":
acceptance_df_2 = acceptance_df_2.drop(columns="Seriendruck-ID")
# Shorten the column names
acceptance_df_2.columns = short_col_names
# Rename the index
acceptance_df_2.index = acceptance_df_2.index.rename('SurveyID')
# -

acceptance_df_1.head(3)

acceptance_df_2.head(3)

# Observe the raw data chunk sizes before merging the chunks:

# +
# Create DataFrame for raw chunks
raw_data_chunks_df = pd.DataFrame(
    {
        "chunk_id": [1,2],
        "observations": [len(acceptance_df_1), len(acceptance_df_2)],
        "all_NAN_rows": [acceptance_df_1.isna().all(axis=1).sum(),
                         acceptance_df_2.isna().all(axis=1).sum()]
    },
)

raw_data_chunks_df = raw_data_chunks_df.set_index("chunk_id")

# Write chunks data frame to csv file
raw_data_chunks_df.to_csv('./data/raw_data/raw_data_chunks.csv',index_label="chunk_id")

#  Check raw data chunk sizes:
raw_data_chunks_df
# -

# Now we merge the chunks, rename the index and check rows consisting of all NaNs:

# +
# Merge the data chunks
acceptance_merged_df = pd.concat([acceptance_df_1,acceptance_df_2])

# Reconstruct the index to continue the numeration of "acceptance_df_1" and rename it:
acceptance_merged_df.index = pd.RangeIndex(
    start=1, stop=1+len(acceptance_merged_df),
    name="SurveyID")

# Write raw data to csv file
acceptance_merged_df.to_csv('./data/raw_data/raw_data_merged.csv',index_label="SurveyID")

acceptance_merged_df[acceptance_merged_df.isna().all(axis=1)]
# -

# We are going to drop the rows with all NaNs. We need to reconstruct the index afterwards to avoid jumps:

# +
# Drop the rows with all NaNs 
acceptance_merged_df = acceptance_merged_df.dropna(how='all')

# Reconstruct the index to avoid jumps/gaps
acceptance_merged_df.index = pd.RangeIndex(
    start=1, stop=len(acceptance_merged_df)+1,
    name="SurveyID")

# Convert to strings to float (int is impossible since there are NAs)
acceptance_merged_df = acceptance_merged_df.astype(float)
# -

# Now we are ready to save the merged data, as well as the new data chunk sizes:

# Write data frame to Excel / csv file
acceptance_merged_df.to_excel('./data/clean_data/data_merged.xlsx')
acceptance_merged_df.to_csv('./data/clean_data/data_merged.csv',index_label="SurveyID")
acceptance_merged_df.head(3) # Check the data

# +
data_chunks_df = raw_data_chunks_df[["observations"]].copy()
data_chunks_df["observations"] = data_chunks_df["observations"] - raw_data_chunks_df["all_NAN_rows"]

# Write chunks data frame to csv file
data_chunks_df.to_csv('./data/clean_data/data_chunks.csv',index_label="chunk_id")
data_chunks_df
# -

# ## Report using Pandas Profiling:

profile = acceptance_merged_df.profile_report(title='Report for the study acceptance data')
profile.to_file(output_file=os.path.join('output','docs','reports','acceptance_report_1.html'));
