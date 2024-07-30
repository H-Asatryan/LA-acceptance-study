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

# **Acceptance Study for Learning Analytics: Specializations and averages**
#
# In this piece of code we compute and plot the averages for all 24 SELAQ features. Then we compare the averages per specialization. The row averages for the question subgroups Data Protection, LA General Functionality, LA Teacher-Related Features are computed, visualized and exported as "row_mean_features.csv". A corresponding HTML report is also generated ("acceptance_report_row_mean_features.html").

# ## Load packages and data

# +
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
# # %matplotlib inline
import seaborn as sns
from functions_and_variables import specializations, specializations_en, specializations_merged, change_height
import ydata_profiling

# Use these lines if you change the loaded functions during your work
# # %load_ext autoreload
# # %autoreload 2
# -

# To load the data from csv, uncomment this line:
acceptance_df = pd.read_csv("./data/clean_data/data_optimized.csv",index_col="SurveyID")
acceptance_imputed_df = pd.read_csv("./data/clean_data/data_imputed.csv",index_col="SurveyID")
data_chunks_df = pd.read_csv('./data/clean_data/data_chunks.csv',index_col="chunk_id")
# acceptance_df.head(3)

# ## Surveyed specializations

# +
# Calculate specilization counts, sort them by their codes
specialization_counts_df = acceptance_imputed_df[["1.1"]].value_counts(
).rename_axis('spec_code').reset_index(name='count')
specialization_counts_df = specialization_counts_df.sort_values("spec_code")
specialization_counts_df["spec_name"] = specializations_en
specialization_counts_df = specialization_counts_df[['spec_code',"spec_name",'count']]

# Export to csv and html
specialization_counts_df.to_html("./output/docs/dataset_info/"+"specialization_counts.html", index=False)
specialization_counts_df = specialization_counts_df.set_index("spec_code")
specialization_counts_df.to_csv("./data/clean_data/specializations_counts.csv", index_label="spec_code")

# View data
specialization_counts_df

# +
# Merging specializations with less than 30 students
specializations_merged_df = specialization_counts_df.copy()

# Merge "Electrical engineering", "Mechanical engineering" and "Mechatronics"
specializations_merged_df.at[3,'count']=specializations_merged_df.at[3,'count'] +specializations_merged_df.loc[5,'count']+specializations_merged_df.loc[6,'count']
# specializations_merged_df.at[3,'count']=specializations_merged_df.loc[3][1] +specializations_merged_df.loc[5][1]+specializations_merged_df.loc[6][1]
specializations_merged_df = specializations_merged_df.drop([5,6])
# specializations_merged_df.at[3,'spec_name']="Electro-mechanical engineering"
specializations_merged_df['spec_name'] = specializations_merged

# Export data (counts) as csv and html
specializations_merged_df.to_csv('./data/clean_data/specializations_merged.csv',index_label="spec_code")
specializations_merged_df.to_html("./output/docs/dataset_info/"+"specializations_merged.html",index=False)
specializations_merged_df
# -

specializations_plot = sns.countplot(y=acceptance_imputed_df["1.1"],
                                     color="blue")
specializations_plot.set_yticklabels(specializations, size=11)
specializations_plot.set_xlabel("Number of students", size=13, labelpad=25)
specializations_plot.set_ylabel(None)
# specializations_plot.get_figure().set_figwidth(8) # use to change the width if needed
change_height(specializations_plot.axes, 0.6)
specializations_plot.get_figure().savefig("./output/docs/dataset_info/" + "specializations_de" +
                                          ".pdf",
                                          format="pdf",
                                          bbox_inches="tight")
specializations_plot.get_figure().savefig("./output/docs/dataset_info/" + "specializations_de" +
                                          ".jpg",
                                          format="jpg",
                                          dpi=150,
                                          bbox_inches="tight")

# sns.set(rc = {'figure.figsize':(5,2)}) # activate to set the size
specializations_en_plot = sns.countplot(y=acceptance_imputed_df["1.1"], color = "blue");
specializations_en_plot.tick_params(axis='x', labelsize=13)
specializations_en_plot.set_xlabel("Number of students", size = 14, labelpad=20);
specializations_en_plot.set_yticklabels(specializations_en, size = 13);
specializations_en_plot.set_ylabel(None);
change_height(specializations_en_plot.axes, 0.7)
specializations_en_fig = specializations_en_plot.get_figure()
# specializations_en_fig = plt.gcf() # alternative to the previous line if matplotlib is imported as plt
# specializations_en_fig.set_size_inches(16,9) # activate to change the size
# specializations_en_fig.set_dpi(300) # activate to change the resolution
specializations_en_fig.savefig(
    "./output/docs/dataset_info/"+"specializations_en"+".pdf", format="pdf", bbox_inches="tight")
specializations_en_fig.savefig( # activate to set larger dpi (default=100)
    "./output/docs/dataset_info/"+"specializations_en"+".jpg", format="jpg", dpi=150, bbox_inches="tight");

# +
# Bar chart for the merged specialization counts
specializations_merged_plot = specializations_merged_df.plot.barh(
    color=["tab:blue"],
    edgecolor='white',
    legend=False,
    width=0.75
)

# Customizing
specializations_merged_plot.invert_yaxis()
specializations_merged_plot.grid(axis="x", linestyle='dotted', linewidth=0.05)
specializations_merged_plot.tick_params(axis='x', labelsize=15)
specializations_merged_plot.set_xlabel('Number of students', size=16, labelpad=18)
# specializations_merged_plot.set_yticklabels(specializations_merged_df["spec_name"], size = 16);
specializations_merged_plot.set_yticklabels(specializations_merged_df["spec_name"], size = 16, fontstretch=0);
specializations_merged_plot.set_ylabel(None);

specializations_merged_plot.get_figure().savefig(
    "./output/docs/dataset_info/"+"specializations_merged"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
specializations_merged_plot.get_figure().savefig(
    "./output/docs/dataset_info/"+"specializations_merged"+".pdf",
    format="pdf", bbox_inches="tight"
);

# + [markdown] heading_collapsed=true
# ## Compute averages

# + hidden=true
# Exclude some categorical columns to compute averages for the rest
acceptance_select_df = acceptance_imputed_df.drop(columns=['1.2', '1.3', '1.4'])
# Select only the SELAQ questions
acceptance_sellaq_df = acceptance_select_df.iloc[:,1:]
# acceptance_sellaq_df.head(2)

# Create portion data frames
acceptance_sellaq_df_1 = acceptance_sellaq_df.iloc[:data_chunks_df["observations"][1],:]
acceptance_sellaq_df_2 = acceptance_sellaq_df.iloc[data_chunks_df["observations"][1]:]

acceptance_global_averages_df = acceptance_sellaq_df.mean().to_frame()
acceptance_global_averages_df.index.name = "Question"
acceptance_global_averages_df.columns = ["Average"]

acceptance_global_averages_df_1 = acceptance_sellaq_df_1.mean().to_frame()
acceptance_global_averages_df_1.index.name = "Question"
acceptance_global_averages_df_1.columns = ["Average_chunk_1"]

acceptance_global_averages_df_2 = acceptance_sellaq_df_2.mean().to_frame()
acceptance_global_averages_df_2.index.name = "Question"
acceptance_global_averages_df_2.columns = ["Average_chunk_2"]

acceptance_global_averages_dfs = acceptance_global_averages_df.join(
    acceptance_global_averages_df_1).join(acceptance_global_averages_df_2)

# acceptance_global_averages_df.head()
acceptance_global_averages_dfs.head(2)

# + hidden=true
# Group by specializations and average
averages_by_specialization = acceptance_select_df.groupby(["1.1"]).mean().T

# Add averages for each specialization to the data frame
acceptance_averages_df = acceptance_global_averages_df.join(averages_by_specialization)

# Add min / max of average marks by specializations to the data frame
acceptance_averages_df = acceptance_averages_df.join(
acceptance_averages_df.agg(['min','max'],axis="columns")
)

# Capitalize min / max columns
acceptance_averages_df = acceptance_averages_df.rename(
    mapper={'min': 'Min', 'max': 'Max'},
    axis='columns'
)

# Round the numbers to 4 decimal places
acceptance_averages_df = round(acceptance_averages_df, 4)
acceptance_averages_df
# -

# ## Export averages

acceptance_averages_df.to_csv('./data/clean_data/data_averages.csv', index=False)

# ## Visualize averages

# We first plot the average grades as a bar chart:

fig_avg_glob = acceptance_averages_df[["Average"]].plot.bar(figsize=(9, 3),
                                                            width=0.35,
                                                            ylim=(0, 7),
                                                            color="cadetblue",
                                                            legend=False,
                                                            rot=70)
# Customizing
fig_avg_glob.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_avg_glob.set_xlabel('Question code', labelpad=15)
fig_avg_glob.set_ylabel('Average grade', labelpad=12)
fig_avg_glob.set_title('Average Grades for SELAQ Questions', pad=13)
fig_avg_glob.get_figure().savefig("./output/docs/dataset_info/"+"average_grades"+".jpg", format="jpg", bbox_inches="tight", dpi = 200);

# +
fig_avg_chunks = acceptance_global_averages_dfs.plot.bar(figsize=(9, 3),
                                                         width=0.5,
                                                         ylim=(0, 7),
                                                         color=["#1f77b4","Crimson", "green"],
#                                                          color=["blue","orange","green"],
                                                         rot=70)

# Customizing
fig_avg_chunks.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_avg_chunks.set_xlabel('Question code', labelpad=15)
fig_avg_chunks.set_ylabel('Average grade', labelpad=12)
fig_avg_chunks.set_title('Average Grades for Data Chunks', pad=13)

# Sort legend titles by name and alter its placement
handles, labels = fig_avg_chunks.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
by_label_sorted = dict(sorted(by_label.items()))
fig_avg_chunks.legend(
    by_label_sorted.values(), by_label_sorted.keys(),
    loc="lower left", bbox_to_anchor = (1, 0.67)
)

fig_avg_chunks.get_figure().savefig(
    "./output/docs/dataset_info/"+"average_grades_chunks"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
);
# -

# Now let us plot a bar chart with min / max values of average grades by specializations, together with global average grades:

# +
# # Version 1 of the graph, better for presentations
# fig_avg_glob_vs_extr = acceptance_averages_df[["Min","Average","Max"]].plot.bar(
#     figsize=(9, 3),
#     width=0.6,
#     ylim=(0, 7),
# #     color=["cadetblue","orange","green"],
#     rot=70
# )

# # Customizing
# fig_avg_glob_vs_extr.grid(axis="y", linestyle='dotted', linewidth=0.4)
# fig_avg_glob_vs_extr.set_xlabel('Question code', labelpad=15)
# fig_avg_glob_vs_extr.set_ylabel('Average grade', labelpad=12)
# fig_avg_glob_vs_extr.set_title('Global Average Grades vs Min/Max of Averages by Specializations', pad=13)

# # Sort legend titles by name and alter its placement
# handles, labels = fig_avg_glob_vs_extr.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# by_label_sorted = dict(sorted(by_label.items()))
# fig_avg_glob_vs_extr.legend(
#     by_label_sorted.values(), by_label_sorted.keys(),
#     loc="lower left", bbox_to_anchor = (1, 0.5)
# )

# fig_avg_glob_vs_extr.get_figure().savefig(
#     "./output/docs/dataset_info/"+"average_min_max_grades_spec"+".pdf",format="pdf", bbox_inches="tight"
# )

# fig_avg_glob_vs_extr.get_figure().savefig(
#     "./output/docs/dataset_info/"+"average_min_max_grades_spec"+".jpg",format="jpg",
#     bbox_inches="tight", dpi = 200
# );

# +
# Paper version (legend above)
fig_avg_glob_vs_extr = acceptance_averages_df[["Min","Average","Max"]].plot.bar(
    figsize=(9, 3),
    width=0.6,
    ylim=(0, 8.1),
#     color=["cadetblue","orange","green"],
    rot=70
)

# Customizing
fig_avg_glob_vs_extr.grid(axis="y", linestyle='dashed', linewidth=0.05)
fig_avg_glob_vs_extr.set_xlabel('Question code', size=15, labelpad=14)
fig_avg_glob_vs_extr.set_ylabel('Average grade', size=15, labelpad=12.5)
fig_avg_glob_vs_extr.set_yticks(range(8))
fig_avg_glob_vs_extr.tick_params(axis='both',
                                 labelsize=11.5
                                )
fig_avg_glob_vs_extr.set_title('Global Average Grades vs Min/Max of Averages by Specializations',
                               size=16,
                               fontstretch=100,
                               pad=13)

# Sort legend titles by name and alter its placement
handles, labels = fig_avg_glob_vs_extr.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
by_label_sorted = dict(sorted(by_label.items()))
fig_avg_glob_vs_extr.legend(
    by_label_sorted.values(),
    by_label_sorted.keys(),
    ncol=3,
    fontsize="large",  # xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller
    loc="center",
    bbox_to_anchor = (0.5, 0.92)
)

fig_avg_glob_vs_extr.get_figure().savefig(
    "./output/docs/dataset_info/"+"average_min_max_grades_spec"+".pdf",format="pdf", bbox_inches="tight"
)

fig_avg_glob_vs_extr.get_figure().savefig(
    "./output/docs/dataset_info/"+"average_min_max_grades_spec"+".jpg",format="jpg",
    bbox_inches="tight", dpi = 200
);
# -

# The above plot shows that for questions 2.1, 2.11, 2.12, 2.21, 2.22 there is a significant difference between grades of different specializations. We can observe this by plotting Max-Min and then try to find the corresponding specializations.

# ### Compare the global averages with the ones for specializations "1, 4" (an example):

# +
fig_avg_glob_vs_1 = acceptance_averages_df[["Average",1,4]].plot.bar(
    figsize=(9, 3),
    width=0.6,
    ylim=(0, 7),
#     color=["cadetblue","orange","green"],
    rot=70
)

# Customizing
fig_avg_glob_vs_1.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_avg_glob_vs_1.set_xlabel('Question code', labelpad=15)
fig_avg_glob_vs_1.set_ylabel('Average grade', labelpad=12)
fig_avg_glob_vs_1.legend(loc="lower left", bbox_to_anchor = (1, 0.5))
fig_avg_glob_vs_1.set_title('Global Average Grades vs Averages of Specializations 1 and 4', pad=13)
fig_avg_glob_vs_1.get_figure().savefig(
    "./output/docs/dataset_info/"+"average_grades_spec_1_4"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
);
# -

# Compare all the specializations with a line plot:

# +
avg_line_plot = acceptance_averages_df.plot(
    figsize=(9, 3),
    rot=70
)

# Customizing
avg_line_plot.grid(linestyle='dotted', linewidth=0.4)
avg_line_plot.set_xlabel('Question code', labelpad=15)
avg_line_plot.set_xticks(range(len(acceptance_averages_df)))
avg_line_plot.set_xticklabels(labels=list(acceptance_averages_df.index))
avg_line_plot.set_ylabel('Average grade', labelpad=12)

# Change line style and width for Average, min and max
for line in avg_line_plot.get_lines():
    if line.get_label() in ["Average","Min","Max"]:
        line.set_linewidth(2.5)
        line.set_linestyle("dashed")
        
avg_line_plot.legend(loc="center left", bbox_to_anchor = (1, 0.5));

avg_line_plot.get_figure().savefig(
    "./output/docs/dataset_info/"+"average_grades_line_plot"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
);


# -

# Note that 1 and 7 seem to be extremal, but they were not significant (due to the low participation numbers)! Hence we plot once more without 1 and 7:

# +
# We first define a function sorting mixed lists containing integers and text
def sort_mixed_list(mixed_list):
    int_part = sorted([i for i in mixed_list if type(i) is int])
    str_part = sorted([i for i in mixed_list if type(i) is str])
    return int_part + str_part

# We specify the columns to be excluded as a set:
columns_to_exclude = {1,7}
# We exclude these columns:
plot_columns = list(set(acceptance_averages_df.columns)-columns_to_exclude)
# plot_columns = [
#     column for column in acceptance_averages_df.columns if column not in columns_to_exclude
# ] # Another way to get plot columns, still unsorted

# We sort the column list:
plot_columns = sort_mixed_list(plot_columns)

# +
avg_line_plot_red = acceptance_averages_df[plot_columns].plot(
    figsize=(9, 3),
    rot=70
)

# Customizing
avg_line_plot_red.grid(linestyle='dotted', linewidth=0.4)
avg_line_plot_red.set_xlabel('Question code', labelpad=15)
avg_line_plot_red.set_xticks(range(len(acceptance_averages_df)))
avg_line_plot_red.set_xticklabels(labels=list(acceptance_averages_df.index))
avg_line_plot_red.set_ylabel('Average grade', labelpad=12)

# Change line style and width for Average, min and max
for line in avg_line_plot_red.get_lines():
    if line.get_label() in ["Average","Min","Max"]:
        line.set_linewidth(2.5)
        line.set_linestyle("dashed")
        
avg_line_plot_red.legend(loc="center left", bbox_to_anchor = (1, 0.5));

avg_line_plot_red.get_figure().savefig(
    "./output/docs/dataset_info/"+"average_grades_red_line_plot"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
);
# -

# We see that 1,2,6 and 7 are the most extremal cases, let us plot them seperately:

# +
avg_line_plot_extr = acceptance_averages_df[["Min","Max",1,2,6,7]].plot(
    figsize=(9, 3),
    rot=70
)

# Customizing
avg_line_plot_extr.grid(linestyle='dotted', linewidth=0.4)
avg_line_plot_extr.set_xlabel('Question code', labelpad=15)
avg_line_plot_extr.set_xticks(range(len(acceptance_averages_df)))
avg_line_plot_extr.set_xticklabels(labels=list(acceptance_averages_df.index))
avg_line_plot_extr.set_ylabel('Average grade', labelpad=12)
avg_line_plot_extr.set_title('Extremal specializations', pad=10)

# Change line style and width for Average, min and max
for line in avg_line_plot_extr.get_lines():
    if line.get_label() in ["Average","Min","Max"]:
        line.set_linewidth(2.5)
        line.set_linestyle("dashed")
        
avg_line_plot_extr.legend(loc="center left", bbox_to_anchor = (1, 0.5));

avg_line_plot_extr.get_figure().savefig(
    "./output/docs/dataset_info/"+"average_grades_extr_line_plot"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
);
# -

# ## The mean squared error (MSE) between average grades per specialization and grade extrema

# Now we are going to confirm our conclusions from the last plot numerically by computing the mean squared errors between the average scores per specialization and the min / max scores.

# +
MSE_from_min = {
    column+1:mean_squared_error(
        acceptance_averages_df[column+1],acceptance_averages_df["Min"]
    ) for column in range(10)
}

MSE_from_max = {
    column+1:mean_squared_error(
        acceptance_averages_df[column+1],acceptance_averages_df["Max"]
    ) for column in range(10)
}

MSE_from_min_sorted = sorted(MSE_from_min.items(), key=lambda x:x[1])
MSE_from_max_sorted = sorted(MSE_from_max.items(), key=lambda x:x[1])

MSE_from_min_sorted_df = pd.DataFrame(MSE_from_min_sorted, columns = ["spec_code","MSE from Min"])
MSE_from_max_sorted_df = pd.DataFrame(MSE_from_max_sorted, columns = ["spec_code","MSE from Max"])
MSE_from_min_sorted_df["spec_name"] = np.take(specializations_en, MSE_from_min_sorted_df["spec_code"]-1)
MSE_from_max_sorted_df["spec_name"] = np.take(specializations_en, MSE_from_max_sorted_df["spec_code"]-1)
MSE_from_min_sorted_df = MSE_from_min_sorted_df[["spec_code","spec_name","MSE from Min"]]
MSE_from_max_sorted_df = MSE_from_max_sorted_df[["spec_code","spec_name","MSE from Max"]]

MSE_from_min_sorted_df.to_html("./output/docs/dataset_info/"+"MSE_from_min.html")
MSE_from_max_sorted_df.to_html("./output/docs/dataset_info/"+"MSE_from_max.html");
# -

MSE_from_min_sorted_df

MSE_from_max_sorted_df

# > Thus, we see that the students from "Mechatronics" (15 students) and "Civil engineering" (137 students) have given the worst grades, whereas the students from "Sustainability" (32 students) and "Architecture" (39 students) have given the best ones.

# ## Row Means

# ### Compute without imputation and visualize

# Define question types
dp_questions_odd = ["2.1", "2.3", "2.5", "2.9", "2.11"]
dp_questions_even = ["2.2", "2.4", "2.6", "2.10", "2.12"]
la_fun_questions_odd = ["2.7", "2.13", "2.15", "2.17", "2.23"]
la_fun_questions_even = ["2.8", "2.14", "2.16", "2.18"] # NOTE: "2.24" is excluded
la_teach_questions_odd = ["2.19", "2.21"]
la_teach_questions_even = ["2.20", "2.22"]
la_questions_odd = ["2.7", "2.13", "2.15", "2.17", "2.19", "2.21", "2.23"]
la_questions_even = ["2.8", "2.14", "2.16", "2.18", "2.20", "2.22"] # NOTE: "2.24" is excluded

# +
# Take row means for each of groups "DP", "LA General", "LA Teacher"
# Instead of "LA General" we can use "LA Functionality" 
desire_group_avg = pd.DataFrame({
    "DP":
    acceptance_df[dp_questions_odd].mean(axis="columns"),
    "LA":
    acceptance_df[la_questions_odd].mean(axis="columns"),
    "LA General":
    acceptance_df[la_fun_questions_odd].mean(axis="columns"),
    "LA Teacher":
    acceptance_df[la_teach_questions_odd].mean(axis="columns"),
})

expectation_group_avg = pd.DataFrame({
    "DP":
    acceptance_df[dp_questions_even].mean(axis="columns"),
    "LA":
    acceptance_df[la_questions_even].mean(axis="columns"),
    "LA General":
    acceptance_df[la_fun_questions_even].mean(axis="columns"),
    "LA Teacher":
    acceptance_df[la_teach_questions_even].mean(axis="columns")
})

# +
# Take column means to get the average desire and expectation
# for "DP", "LA General" and "LA Teacher"
desire_averages = desire_group_avg.mean()
expectation_averages = expectation_group_avg.mean()

# Create DataFrame of desire / expectation
desire_expect_averages_df = pd.DataFrame({
    "Desire": desire_averages,
    "Expectation": expectation_averages
})

desire_expect_averages_df

# +
fig_desire_expec_avg = desire_expect_averages_df.loc[["DP","LA"],:].plot.bar(
    figsize=(2.5, 2),
    width=0.5,
#     xlim=(0, 2),
    rot=0,
    linewidth=2,
    edgecolor='white',
    color=['#4F81BD', '#C0504D'] # "Crimson","SlateBlue","LightCoral","cadetblue","green"
)

# Customizing
fig_desire_expec_avg.grid(axis="y", linestyle='dotted', linewidth=0.35)
# fig_desire_expec_avg.set_xlabel('Question groups', labelpad=12)
fig_desire_expec_avg.set_ylabel('Average grade', size = "medium", labelpad=20)
fig_desire_expec_avg.tick_params(axis='x', labelsize="small")
fig_desire_expec_avg.set_yticks(
    np.arange(0, 6.5, 0.5),
    size = "small",
    labels=np.vstack([np.arange(0, 7, 1), [""]*7]).ravel('F')[:-1]
)

# Alter legend's placement
fig_desire_expec_avg.legend(
    loc="center left",
    fontsize = "small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (1, 0.85)
)

fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_1"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_1"+".pdf",format="pdf", bbox_inches="tight"
);

# +
# Paper version
fig_desire_expec_avg = desire_expect_averages_df.loc[["DP","LA"],:].plot.bar(
    figsize=(2.75, 2),
    width=0.5,
#     xlim=(0, 2),
    ylim=(0, 7.7),
    rot=0,
    linewidth=2,
    edgecolor='white',
    color=['#4F81BD', '#C0504D'] # "Crimson","SlateBlue","LightCoral","cadetblue","green"
)

# Customizing
fig_desire_expec_avg.grid(axis="y", linestyle='dotted', linewidth=0.1)
# fig_desire_expec_avg.set_xlabel('Question groups', labelpad=12)
fig_desire_expec_avg.set_ylabel('Average grade', size = "medium", labelpad=15)
fig_desire_expec_avg.tick_params(axis='x', labelsize="small")
# fig_desire_expec_avg.set_yticks(
#     np.arange(0, 6.5, 0.5),
#     size = "small",
#     labels=np.vstack([np.arange(0, 7, 1), [""]*7]).ravel('F')[:-1]
# )


# Alter legend's placement
fig_desire_expec_avg.legend(
    loc="center",
    fontsize = "small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (0.5, 0.9),
    ncol=2
)

fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_1"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_1"+".pdf",format="pdf", bbox_inches="tight"
);

# +
fig_desire_expec_avg = desire_expect_averages_df.loc[["DP","LA General","LA Teacher"],:].plot.bar(
    figsize=(3.5, 2),
    width=0.5,
#     xlim=(0, 2),
    rot=0,
    linewidth=2,
    edgecolor='white',
    color=['#4F81BD', '#C0504D'] # "Crimson","SlateBlue","LightCoral","cadetblue","green"
)

# Customizing
fig_desire_expec_avg.grid(axis="y", linestyle='dotted', linewidth=0.35)
# fig_desire_expec_avg.set_xlabel('Question groups', labelpad=12)
fig_desire_expec_avg.set_ylabel('Average grade', size = "medium", labelpad=20)
fig_desire_expec_avg.tick_params(axis='x', labelsize="small")
fig_desire_expec_avg.set_yticks(
    np.arange(0, 6.5, 0.5),
    size = "small",
    labels=np.vstack([np.arange(0, 7, 1), [""]*7]).ravel('F')[:-1]
)

# Alter legend's placement
fig_desire_expec_avg.legend(
    loc="center left",
    fontsize = "small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (1, 0.85)
)

fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_2"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_2"+".pdf",format="pdf", bbox_inches="tight"
);

# +
# Paper version

fig_desire_expec_avg = desire_expect_averages_df.loc[["DP","LA General","LA Teacher"],:].plot.bar(
    figsize=(3.5, 2),
    width=0.5,
#     xlim=(0, 2),
    ylim=(0, 7.7),
    rot=0,
    linewidth=2,
    edgecolor='white',
    color=['#4F81BD', '#C0504D'] # "Crimson","SlateBlue","LightCoral","cadetblue","green"
)

# Customizing
fig_desire_expec_avg.grid(axis="y", linestyle='dashed', linewidth=0.1)
# fig_desire_expec_avg.set_xlabel('Question groups', labelpad=12)
fig_desire_expec_avg.set_ylabel('Average grade', size = "medium", labelpad=15)
fig_desire_expec_avg.tick_params(axis='x', labelsize="small")
# fig_desire_expec_avg.set_yticks(
#     np.arange(0, 6.5, 0.5),
#     size = "small",
#     labels=np.vstack([np.arange(0, 7, 1), [""]*7]).ravel('F')[:-1]
# )

# Alter legend's placement
fig_desire_expec_avg.legend(
    loc="center",
    fontsize = "small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (0.5, 0.9),
    ncol=2
)

fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_2"+".jpg",format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg.get_figure().savefig(
    "./output/docs/dataset_info/"+"desire_expec_avg_2"+".pdf",format="pdf", bbox_inches="tight"
);

# +
# Create a dataframe from row averages for the
# groups "DP", "LA General" and "LA Teacher"
row_mean_features_df = pd.concat([desire_group_avg, expectation_group_avg], axis=1)

# Rename columns, step 1: add Desire/Expectation to the names
feature_colnames = np.array(row_mean_features_df.columns)
feature_colnames[:4] = feature_colnames[:4] +" Desire"
feature_colnames[-4:] = feature_colnames[-4:] +" Expectation"
row_mean_features_df.columns = feature_colnames

# Rename columns, step 2: reorder columns alternatingly
feature_col_order = ["None"]*len(feature_colnames)
feature_col_order[::2] = feature_colnames[:4]
feature_col_order[1::2] = feature_colnames[-4:]
row_mean_features_df = row_mean_features_df[feature_col_order]
# -

# ### Missing Data in Row Means

# Number of missing data
print("{} Samples with missing data.".format(
    row_mean_features_df.isna().any(axis='columns').sum()
))

# In the data frame "row_mean_features_df" we obtain some NA rows, because we have missing values for some question types. We get 18 NA rows (27 NA rows if we use not optimized acceptance data!) and we will compare their NA shares to the ones in the data set "acceptance_df":

na_comparison_df = row_mean_features_df.copy()
na_comparison_df["na_share_acceptance"] = acceptance_df.isnull().sum(axis=1)/acceptance_df.shape[1]
na_comparison_df["na_share_row_means"] = row_mean_features_df.isnull().sum(axis=1)/row_mean_features_df.shape[1]
na_comparison_df = na_comparison_df.iloc[:,-2:]
na_comparison_df = round(100*na_comparison_df)
na_comparison_df

na_comparison_df[na_comparison_df["na_share_acceptance"]>30]

# Note that imputing "acceptance_df" rows with more than 30% NAs would reduce NAs in "row_mean_features_df" (see "030_reduce_impute_data.py").

na_comparison_df[na_comparison_df["na_share_row_means"]>0][na_comparison_df["na_share_acceptance"]<30]

na_acceptance = na_comparison_df.sort_values("na_share_acceptance",ascending=False)
na_acceptance = na_acceptance[na_acceptance["na_share_acceptance"]>0]
print(len(na_acceptance), "rows with NAs in 'acceptance' data frame:")
na_acceptance

na_row_means = na_comparison_df.sort_values("na_share_row_means",ascending=False)
na_row_means = na_row_means[na_row_means["na_share_row_means"]>0]
print(len(na_row_means), "rows with NAs in 'row_means' data frame:")
na_row_means

# We may drop those NAs:

row_mean_features_reduced_df = row_mean_features_df.dropna()
row_mean_features_reduced_df.head(2)

# Export the DataFrame as csv
row_mean_features_reduced_df.to_csv('./data/clean_data/row_mean_features_reduced.csv',index_label="SurveyID")

# ### Row means based on imputed acceptance data and its report

# +
# Take row means for each of groups "DP", "LA General", "LA Teacher"
# Instead of "LA General" we can use "LA Functionality" 
desire_group_avg_imputed = pd.DataFrame({
    "DP":
    acceptance_imputed_df[dp_questions_odd].mean(axis="columns"),
    "LA":
    acceptance_imputed_df[la_questions_odd].mean(axis="columns"),
    "LA General":
    acceptance_imputed_df[la_fun_questions_odd].mean(axis="columns"),
    "LA Teacher":
    acceptance_imputed_df[la_teach_questions_odd].mean(axis="columns"),
})

expectation_group_avg_imputed = pd.DataFrame({
    "DP":
    acceptance_imputed_df[dp_questions_even].mean(axis="columns"),
    "LA":
    acceptance_imputed_df[la_questions_even].mean(axis="columns"),
    "LA General":
    acceptance_imputed_df[la_fun_questions_even].mean(axis="columns"),
    "LA Teacher":
    acceptance_imputed_df[la_teach_questions_even].mean(axis="columns")
})

# +
# Create a dataframe from row averages for the
# groups "DP", "LA General" and "LA Teacher"
row_mean_feat_imputed_df = pd.concat([desire_group_avg_imputed, expectation_group_avg_imputed], axis=1)

# Rename columns, step 1: add Desire/Expectation to the names
row_mean_feat_imputed_df.columns = feature_colnames

# Rename columns, step 2: reorder columns alternatingly
row_mean_feat_imputed_df = row_mean_feat_imputed_df[feature_col_order]
# -

# Number of missing data
print("{} Samples with missing data.".format(
    row_mean_feat_imputed_df.isna().any(axis='columns').sum()
))

# Export the DataFrame as csv
row_mean_feat_imputed_df.to_csv('./data/clean_data/row_mean_features.csv',index_label="SurveyID")

# Report using ydata profiling:

profile = row_mean_features_df.drop(columns=["LA Desire","LA Expectation"]).profile_report(title='Report for the acceptance study based on row mean features')
profile.to_file(output_file='./output/docs/reports/acceptance_report_2_row_mean_features.html');
