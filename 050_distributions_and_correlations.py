# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Acceptance Study: Visualizing distributions and correlations
# This piece of code visualizes the imputed data. We plot and export all the histograms, a heatmap and scatterplots for most correlated variables.

# %% [markdown]
# ### Import packages and data

# %%
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# from jupyterthemes import jtplot # dark mode for graphs
# jtplot.style() # dark mode for graphs
from math import ceil
from functions_and_variables import plot_df_histograms, specializations_en

# Use these lines if you change the loaded functions during your work
# %load_ext autoreload
# %autoreload 2

# %%
acceptance_df = pd.read_excel('./data/clean_data/data_imputed.xlsx',index_col="SurveyID")
# We need the imputed data for some tasks below!

# %% [markdown]
# ### Histograms for all pre-processed columns

# %%
suptitle = f"All specializations:  {len(acceptance_df)} students"
plot_df_histograms(acceptance_df,output_name="docs/dataset_info/histograms_all", suptitle = suptitle)

# %%
# acceptance_df.plot(kind='hist',subplots=True, layout=(6, 3));
# plot_histograms(acceptance_df) # another function from "functions_and_variables"
# fig = plt.figure()
# fig.write_html(file_name)

# %% [markdown]
# ### Lower and upper grades

# %%
acceptance_df_ext = acceptance_df.copy()

# Column of average values (for 2.1-2:20)
acceptance_df_ext['mean'] = round(acceptance_df_ext.iloc[:,4:].mean(axis=1), 2)

# %%
acceptance_df_lower = acceptance_df[round(acceptance_df_ext["mean"])<=2]
acceptance_df_upper = acceptance_df[round(acceptance_df_ext["mean"])>=6]
acceptance_df_lower

# %%
plot_df_histograms(acceptance_df_lower, output_name="docs/dataset_info/histograms_lower")

# %%
specializations_plot = sns.countplot(y=acceptance_df_lower["1.1"], color = "blue");
yticklabels_plttext = specializations_plot.get_yticklabels()
yticklabel_numbers = [int(item.get_text())-1 for item in yticklabels_plttext]
yticklabel_descriptions = [specializations_en[idx] for idx in yticklabel_numbers]
specializations_plot.set_yticklabels(yticklabel_descriptions, size = 11);
specializations_plot.set_xlabel("Number of students who graded 1 or 2", size = 13, labelpad=21);
specializations_plot.set_ylabel(None);
specializations_plot.get_figure().set_figheight(3) # use to change the height if needed
specializations_plot.get_figure().savefig("./output/docs/dataset_info/"+"students_graded_1_or_2"+".pdf", format="pdf", bbox_inches="tight");

# %% [markdown]
# >So, the students of *Civil engineering (Bauingenieurwesen)* are less satisfied than other ones!

# %%
plot_df_histograms(acceptance_df_upper, output_name="docs/dataset_info/histograms_upper")

# %% [markdown]
# ## Correlations

# %% [markdown]
# A beautiful way of looking at this data is the use of heatmaps:

# %%
corr_matrix = acceptance_df.iloc[:,1:].corr()

# Mask the diagonal and low correlations
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.abs(corr_matrix) < 0.3] = True
mask[np.triu_indices_from(mask)] = True

# %%
plt.figure(figsize=(16,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", 
xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
cmap= "bwr");
plt.savefig("./output/docs/dataset_info/"+"heatmap_all"+".pdf", format="pdf", bbox_inches="tight");
plt.savefig("./output/docs/dataset_info/"+"heatmap_all"+".jpg", format="jpg", bbox_inches="tight");

# %%
plt.figure(figsize=(16,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", mask=mask,
xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
cmap= "bwr");
plt.savefig("./output/docs/dataset_info/"+"heatmap_all_above_0_3"+".pdf", format="pdf", bbox_inches="tight");
plt.savefig("./output/docs/dataset_info/"+"heatmap_all_above_0_3"+".jpg", format="jpg", bbox_inches="tight");

# %% [markdown]
# ### Scatterplots for most correlated variables

# %%
fig, axs = plt.subplots(2, 3, figsize=(9, 7))
plt.subplots_adjust(bottom=0.1,wspace=0.4,hspace=0.35)
fig.suptitle('The most correlated variables', fontsize=15)
ax_list = [["2.7", "2.13"], ["2.14", "2.18"], ["2.8", "2.14"], ["2.5", "2.9"], ["2.3", "2.9"], ["2.1", "2.5"]]

for idx in range(len(ax_list)):
    row = idx//3
    col = idx-row*3
    sns.scatterplot(data=acceptance_df, x=ax_list[idx][0], y=ax_list[idx][1], ax=axs[row, col])

plt.savefig("./output/docs/dataset_info/"+"scatterplot_most_correl"+".pdf", format="pdf", bbox_inches="tight");
plt.show()

# %% [markdown]
# We observe that the scatterplots do not visualize the correlations we saw on the heatmap because of the discreteness of the values.
