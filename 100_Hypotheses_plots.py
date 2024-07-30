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

# # Acceptance Study: Some Hypotheses and Plots

# Based on the previous studies, especially histograms per cluster (see "090_PCA_and_Kmeans.py"), we revealed the following interesting questions:
# - Test if $(2.8\to\max)\implies(\text{all}\to\max)$ / filter by `2.8 = max` and test if `all = max`.
# - Test if $(2.19\to\min)\implies(2.21, 2.22\to\min)$ / filter by `2.19 = min` and test if `2.21, 2.22 = min`.
# - Test if $(2.1\to\min)\implies(\text{all away from mean})$ / filter by `2.1 = min` and test if `all << max`.
# - Test if $(2.6\to\min)\implies(\text{all away from mean})$ / filter by `2.6 = min` and test if `all << max`.
# - Lowest grade ($\leq 1$) frequencies / bar chart.

# ## Import packages and data

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # to copy plotly objects etc
import numpy as np
# import matplotlib.pyplot as plt
# # %matplotlib inline
# import seaborn as sns

# Load imputed data
acceptance_imputed_df = pd.read_excel('./data/clean_data/data_imputed.xlsx',index_col="SurveyID")
acceptance_averages_df = pd.read_csv('./data/clean_data/data_averages.csv')

# ## Hypothesis 1: $(2.8\to\max)\implies(\text{all}\to\max)$

# +
acceptance_subset_1 = acceptance_imputed_df[acceptance_imputed_df["2.8"] > 6.1].mean()
acceptance_subset_1 = acceptance_subset_1.to_frame().loc["2.1":,:].squeeze()

# Change the color for the bar "2.8" to "Crimson" (red), the default is "#1f77b4" 
# Uncomment the following command to check the defaults
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# Other good choices: "SteelBlue", "#0489B1", "#045FB4"
column_colors_1 = 7*["#1f77b4"]+["Crimson"]+16*["#1f77b4"]
plot_1 = acceptance_subset_1.plot.bar(color=column_colors_1);
plot_1.set_title("Effects of high grades for 2.8")

# Plot total averages on the same graph:
acceptance_averages_df["Average"].plot.line(color="tab:orange",rot=70);

# Extract legend handles and customize the legend:
handles, labels = plot_1.get_legend_handles_labels()
plot_1.legend(
    [handles[1].patches[7],
     handles[1].patches[0],
     handles[0]],
    ["Filter column (larger than 6.1)",
     "Remaining columns after filtration",
     "Total average"],
    loc="lower left", bbox_to_anchor = (1, 0.8)
)

plot_1_fig = plot_1.get_figure()
plot_1_fig.savefig(
    "./output/extra/hypotheses/"+"high_grades_28"+".pdf",
    format="pdf", bbox_inches="tight"
)
plot_1_fig.savefig( # activate to set larger dpi (default=100)
    "./output/extra/hypotheses/"+"high_grades_28"+".jpg",
    format="jpg", dpi=150, bbox_inches="tight"
);

# +
# Paper version (legend above)
acceptance_subset_1 = acceptance_imputed_df[acceptance_imputed_df["2.8"] > 6.1].mean()
acceptance_subset_1 = acceptance_subset_1.to_frame().loc["2.1":,:].squeeze()

# Change the color for the bar "2.8" to "Crimson" (red), the default is "#1f77b4" 
# Uncomment the following command to check the defaults
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# Other good choices: "SteelBlue", "#0489B1", "#045FB4"
column_colors_1 = 7*["#1f77b4"]+["Crimson"]+16*["#1f77b4"]
plot_1 = acceptance_subset_1.plot.bar(
    figsize=(6,3),
    width=0.55,
    color=column_colors_1,
    ylim=(0, 9.5)
);
# plot_1.set_title("Effects of high grades for 2.8")
plot_1.set_yticks(np.arange(0, 8, 2))

# Plot total averages on the same graph:
acceptance_averages_df["Average"].plot.line(color="tab:orange",rot=70);

# Extract legend handles and customize the legend:
handles, labels = plot_1.get_legend_handles_labels()
plot_1.legend(
    [handles[1].patches[7],
     handles[1].patches[0],
     handles[0]],
    ["Filter column (larger than 6.1)",
     "Remaining columns after filtration"+2*" ",
     "Total average"],
    loc="center",
    ncol=2,
    bbox_to_anchor = (0.5, 0.89)
)

plot_1_fig = plot_1.get_figure()
plot_1_fig.savefig(
    "./output/extra/hypotheses/"+"high_grades_28"+".pdf",
    format="pdf", bbox_inches="tight"
)
plot_1_fig.savefig( # activate to set larger dpi (default=100)
    "./output/extra/hypotheses/"+"high_grades_28"+".jpg",
    format="jpg", dpi=150, bbox_inches="tight"
);
# -

# So the hypothesis is true!

# ## Hypothesis 2: $(2.19\to\min)\implies(2.21, 2.22\to\min)$

# +
acceptance_subset_2 = acceptance_imputed_df[acceptance_imputed_df["2.19"] < 1.01].mean()
acceptance_subset_2 = acceptance_subset_2.to_frame().loc[["2.19","2.21","2.22"],:].squeeze()
column_colors_2 = ["Crimson"]+2*["#1f77b4"]
plot_2 = acceptance_subset_2.plot.bar(
    figsize=(1.5,4),
    width=0.35,
#     xticks=[0,1,2],
    color = column_colors_2
)
plot_2.set_title(
    "Effects of low grades for 2.19 on 2.21, 2.22",
    x=1.7, style='italic', fontsize=10, pad=16
)
# ax.set_title('Manual y', y=1.0, pad=-14)

# Plot total averages on the same graph:
acceptance_avg_19_21 = acceptance_averages_df.iloc[18:21,0:1]
acceptance_avg_19_21 = acceptance_avg_19_21.reset_index()["Average"]
acceptance_avg_19_21.plot.line(
#     ax=plot_2,
    color="green",
#     xticks=[0,1,2],
    rot=70
);

plot_2.set_xticklabels(["2.19","2.21","2.22"])

# Extract legend handles and customize the legend:
handles, labels = plot_2.get_legend_handles_labels()
plot_2.legend(
    [handles[1].patches[0],
     handles[1].patches[1],
     handles[0]],
    ["Filter column (smaller than 1.01)",
     "Remaining columns after filtration",
     "Total average"],
    loc="lower left", bbox_to_anchor = (1, 0.76)
)

plot_2_fig = plot_2.get_figure()
plot_2_fig.savefig(
    "./output/extra/hypotheses/"+"low_grades_219"+".pdf",
    format="pdf", bbox_inches="tight"
)
plot_2_fig.savefig( # activate to set larger dpi (default=100)
    "./output/extra/hypotheses/"+"low_grades_219"+".jpg",
    format="jpg", dpi=150, bbox_inches="tight"
);
# -

# So the hypothesis is true!

# ## Hypothesis 3: $(2.1\to\min)\implies(\text{all away from mean})$

# +
acceptance_subset_3 = acceptance_imputed_df[acceptance_imputed_df["2.1"] < 1.01].mean()
acceptance_subset_3 = acceptance_subset_3.to_frame().loc["2.1":,:].squeeze()
column_colors_3 = ["Crimson"]+23*["#1f77b4"]
plot_3 = acceptance_subset_3.plot.bar(color=column_colors_3);
plot_3.set_title("Effects of low grades for 2.1",pad=9)

# Plot total averages on the same graph:
acceptance_averages_df["Average"].plot.line(color="green",rot=70);

# Extract legend handles and customize the legend:
handles, labels = plot_3.get_legend_handles_labels()
plot_3.legend(
    [handles[1].patches[0],
     handles[1].patches[7],
     handles[0]],
    ["Filter column (smaller than 1.01)",
     "Remaining columns after filtration",
     "Total average"],
    loc="lower left", bbox_to_anchor = (1, 0.8)
)

plot_3_fig = plot_3.get_figure()
plot_3_fig.savefig(
    "./output/extra/hypotheses/"+"low_grades_21"+".pdf",
    format="pdf", bbox_inches="tight"
)
plot_3_fig.savefig( # activate to set larger dpi (default=100)
    "./output/extra/hypotheses/"+"low_grades_21"+".jpg",
    format="jpg", dpi=150, bbox_inches="tight"
);
# -

# So the hypothesis is true!

# ## Hypothesis 4: $(2.6\to\min)\implies(\text{all away from mean})$

# +
acceptance_subset_4 = acceptance_imputed_df[acceptance_imputed_df["2.6"] < 1.01].mean()
acceptance_subset_4 = acceptance_subset_4.to_frame().loc["2.1":,:].squeeze()
column_colors_4 = 5*["#1f77b4"]+["Crimson"]+18*["#1f77b4"]
plot_4 = acceptance_subset_4.plot.bar(color=column_colors_4);
plot_4.set_title("Effects of low grades for 2.6",pad=9)

# Plot total averages on the same graph:
acceptance_averages_df["Average"].plot.line(color="green",rot=70);

# Extract legend handles and customize the legend:
handles, labels = plot_4.get_legend_handles_labels()
plot_4.legend(
    [handles[1].patches[0],
     handles[1].patches[5],
     handles[0]],
    ["Filter column (smaller than 1.01)",
     "Remaining columns after filtration",
     "Total average"],
    loc="lower left", bbox_to_anchor = (1, 0.8)
)

plot_4_fig = plot_4.get_figure()
plot_4_fig.savefig(
    "./output/extra/hypotheses/"+"low_grades_26"+".pdf",
    format="pdf", bbox_inches="tight"
)
plot_4_fig.savefig( # activate to set larger dpi (default=100)
    "./output/extra/hypotheses/"+"low_grades_26"+".jpg",
    format="jpg", dpi=150, bbox_inches="tight"
);

# +
# Paper version (legends above)

acceptance_subset_4 = acceptance_imputed_df[acceptance_imputed_df["2.6"] < 1.01].mean()
acceptance_subset_4 = acceptance_subset_4.to_frame().loc["2.1":,:].squeeze()
column_colors_4 = 5*["#1f77b4"]+["Crimson"]+18*["#1f77b4"]
plot_4 = acceptance_subset_4.plot.bar(
    figsize=(6,3),
    width=0.55,
    color=column_colors_4,
    ylim=(0, 9)
);
# plot_4.set_title("Effects of low grades for 2.6",pad=9)
plot_4.set_yticks(np.arange(0, 8, 2))

# Plot total averages on the same graph:
acceptance_averages_df["Average"].plot.line(color="tab:orange",rot=70);

# Extract legend handles and customize the legend:
handles, labels = plot_4.get_legend_handles_labels()
plot_4.legend(
    [handles[1].patches[5],
     handles[1].patches[0],
     handles[0]],
    ["Filter column (smaller than 1.01)",
     "Remaining columns after filtration"+2*" ",
     "Total average"],
    loc="center",
    ncol=2,
    bbox_to_anchor = (0.5, 0.89)
)

plot_4_fig = plot_4.get_figure()
plot_4_fig.savefig(
    "./output/extra/hypotheses/"+"low_grades_26"+".pdf",
    format="pdf", bbox_inches="tight"
)
plot_4_fig.savefig( # activate to set larger dpi (default=100)
    "./output/extra/hypotheses/"+"low_grades_26"+".jpg",
    format="jpg", dpi=150, bbox_inches="tight"
);
# -

# So the hypothesis is true!

# ## Animated Plots for all Hypotheses

# +
# Data frames for upper/lower "constraints"
# We average "acceptance_imputed_df", e.g., by considering all
# the items with a grade >6.1 for the question "2.1". Then we
# compare the obtained averages to the total ones.

lower_marks_list = []
upper_marks_list = []
for col_name in acceptance_imputed_df.columns[4:]:
    acceptance_subset_up = acceptance_imputed_df[
        acceptance_imputed_df[col_name] > 6.1].mean()
    acceptance_subset_up = round(
        acceptance_subset_up.to_frame().loc["2.1":, :].squeeze(), 3)
    acceptance_subset_up_df = acceptance_subset_up.to_frame(
        name="values").reset_index().rename(columns={'index': 'questions'})
    acceptance_subset_up_df = pd.concat([
        acceptance_subset_up_df,
        pd.Series(len(acceptance_subset_up) * [col_name])
    ],
        axis=1)
    upper_marks_list.append(acceptance_subset_up_df)

    acceptance_subset_low = acceptance_imputed_df[
        acceptance_imputed_df[col_name] < 1.1].mean()
    acceptance_subset_low = round(
        acceptance_subset_low.to_frame().loc["2.1":, :].squeeze(), 3)
    acceptance_subset_low_df = acceptance_subset_low.to_frame(
        name="values").reset_index().rename(columns={'index': 'questions'})
    acceptance_subset_low_df = pd.concat([
        acceptance_subset_low_df,
        pd.Series(len(acceptance_subset_low) * [col_name])
    ],
                                         axis=1)
    lower_marks_list.append(acceptance_subset_low_df)

upper_marks_list_df = pd.concat(
    upper_marks_list, axis=0).reset_index().rename(columns={0: '6.1 < column'})
upper_marks_list_df['constraint_var'] = upper_marks_list_df.apply(
    lambda row: row[1] == row[3], axis=1)

lower_marks_list_df = pd.concat(
    lower_marks_list, axis=0).reset_index().rename(columns={0: '1.1 > column'})
lower_marks_list_df['constraint_var'] = lower_marks_list_df.apply(
    lambda row: row[1] == row[3], axis=1)
# -

upper_marks_list_df

fig_up = px.bar(
    data_frame=upper_marks_list_df,
    x="questions",
    y="values",
    animation_frame="6.1 < column",
    color="constraint_var",
    color_discrete_sequence=["Crimson", "#1f77b4"],
    category_orders={'questions': upper_marks_list_df.questions},
    title="Effects of high grades for one question",
    width=840,
    height=500,
    labels={  # label replacement mask
        "questions": "Questions",
        "values": "Grades",
        "constraint_var": "Constrained",
        "6.1 < column": "Constrained"
    })
# fig_up["layout"].pop("updatemenus") # drop animation play buttons
fig_up.update_layout(
    title_x=0.5,
    title_y=.935,
    #     xaxis_title="Questions", # another way to change x, y labels
    #     yaxis_title="Grades",
    margin=dict(t=60))
fig_up.update_xaxes(tickangle=-45)
fig_up.write_html("./output/extra/hypotheses/upper_constraints_0.html",auto_play=False)
# fig_up.show()

# +
# lower_marks_list_df
# -

fig_low = px.bar(
    data_frame=lower_marks_list_df,
    x="questions",
    y="values",
    animation_frame="1.1 > column",
    color="constraint_var",
    color_discrete_sequence=["Crimson", "#1f77b4","green"],
    category_orders={'questions': lower_marks_list_df.questions},
    title="Effects of low grades for one question",
    width=840,
    height=500,
    #     opacity=0.5,
    labels={  # label replacement mask
        "questions": "Questions",
        "values": "Grades",
        "constraint_var": "Constrained",
        "1.1 > column": "Constrained"
    })
fig_low["layout"].pop("updatemenus") # drop animation play buttons
fig_low.update_layout(
    title_x=0.5,
    title_y=.935,
    #     xaxis_title="Questions", # another way to change x, y labels
    #     yaxis_title="Grades",
    margin=dict(t=60))
fig_low.update_xaxes(tickangle=-45)
# fig_low.update_xaxes(tickangle=-45, tickvals=np.arange(0, 24)) # no advantage
fig_low.write_html("./output/extra/hypotheses/lower_constraints_0.html",auto_play=False)
# fig_low.show()

# ## New Animated Layered Plots

# The original averages
fig_origin = px.line(
    data_frame=acceptance_averages_df.set_index(
        acceptance_imputed_df.columns[4:]),
    color_discrete_sequence=["brown"],
    y="Average",
    line_shape="spline"
)
fig_origin.update_traces(showlegend = True, name="Global average");
# fig_origin

# Copy the plotly figure
fig_up_merged = go.Figure(fig_up)
# This adds a line plot over frames
fig_up_merged.add_trace(fig_origin.data[0])
fig_up_merged["layout"].pop("updatemenus") # drop animation play buttons
fig_up_merged.write_html("./output/extra/hypotheses/upper_constraints.html",auto_play=False)
# To save the first frame as an image, uncomment the following line
# You can save other frames directly from html
# fig_up_merged.write_image(file="./output/upper_constraints.jpg", format="jpg", scale=3) # scale > 1 improves the resolution
fig_up_merged.show()
# fig_up_merged.show('browser') # open in browser

# Copy the plotly figure
fig_low_merged = go.Figure(fig_low)
# This adds a line plot over frames
fig_low_merged.add_trace(fig_origin.data[0])
fig_low_merged["layout"].pop("updatemenus") # drop animation play buttons
fig_low_merged.write_html("./output/extra/hypotheses/lower_constraints.html",auto_play=False)
fig_low_merged.show()
# fig_up_merged.show('browser') # open in browser
