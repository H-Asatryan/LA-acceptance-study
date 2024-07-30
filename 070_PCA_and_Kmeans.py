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

# # Acceptance Study: PCA and K-Means Clustering

# ## Import packages and data

# +
import pandas as pd
import numpy as np
from math import ceil

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
# pip install -U kaleido # to export plotly figures as static images
from functions_and_variables import specializations_en,specializations_merged,plot_df_histograms

# +
# Load data
acceptance_imputed_df = pd.read_excel('./data/clean_data/data_imputed.xlsx',index_col="SurveyID")
row_mean_features_df = pd.read_csv("./data/clean_data/row_mean_features.csv",index_col="SurveyID")

# Drop the superfluous columns
acceptance_imputed_df = acceptance_imputed_df.drop(
    acceptance_imputed_df.iloc[:, 1:4], axis=1)

# Exclude the categorical (integer) column,
# corresponding to specializations
acceptance_num = acceptance_imputed_df.iloc[:, 1:]
# -

# ## PCA

# ### Scaling

# Scaling numerical features for PCA
scaler = RobustScaler()
acceptance_scaled = pd.DataFrame(scaler.fit_transform(acceptance_num),
                                 columns=acceptance_num.columns)
# Note: We use robust scalers, because we did not remove outliers.

# ### Heatmaps for the scaled data (run on demand)

# +
corr_matrix = acceptance_scaled.corr()

# Mask the diagonal and low correlations
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.abs(corr_matrix) < 0.3] = True
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(16,10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f",xticklabels=corr_matrix.columns,
    yticklabels=corr_matrix.columns,cmap= "coolwarm"
);
plt.savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"heatmap_scaled_all"+".pdf",
    format="pdf",
    bbox_inches="tight"
);
plt.savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"heatmap_scaled_all"+".jpg",
    format="jpg",
    bbox_inches="tight"
);
# -

plt.figure(figsize=(16,10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", mask=mask, xticklabels=corr_matrix.columns,
    yticklabels=corr_matrix.columns,cmap= "bwr"
);
plt.savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"heatmap_scaled_all_above_0_3"+".pdf",
    format="pdf", bbox_inches="tight"
);
plt.savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"heatmap_scaled_all_above_0_3"+".jpg",
    format="jpg", bbox_inches="tight"
);

# After comparing these correlations to those without scaling / imputing we observe that the values here are slightly smaller that the old ones. For the largest values we have a difference of about `0.01`.

# ### PCA initialization

pca = PCA()
pca.fit(acceptance_scaled); # find all the principal components (no target!)

# Now we project our `acceptance_scaled` dataset onto the new space with the number of principal components we decided to keep. We name it `acceptance_proj`

# +
n_pcs = 3 # Threshold PCA
pca = PCA(n_components=n_pcs, whiten=True)
pca.fit(acceptance_scaled)
acceptance_proj = pd.DataFrame(pca.transform(acceptance_scaled))
# acceptance_proj.head(2)

# This command also prevents NaN values after column assignment
acceptance_proj = acceptance_proj.set_index(acceptance_imputed_df.index)
# Naming PCA features for plotly visualizations
acceptance_proj.columns=["pr_PC1","pr_PC2","pr_PC3"]
# -

# ### Cumulated explained variance and the elbow method (run on demand)

# The Elbow Method for PCA
with plt.style.context('seaborn-deep'):
    # figsize
    plt.figure(figsize=(10,6))
    # getting axes
    ax = plt.gca()
    # plotting
    explained_variance_ratio_cumulated = np.cumsum(pca.explained_variance_ratio_)
    x_axis_ticks = np.arange(1,explained_variance_ratio_cumulated.shape[0]+1)
    ax.plot(x_axis_ticks,explained_variance_ratio_cumulated,
            label="cumulated variance ratio",color="purple",linestyle=":",marker="D",markersize=10)
    # customizing
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulated explained variance (ratio)')
    ax.legend(loc="upper left")
    ax.set_title('The Elbow Method')
    ax.set_xticks(x_axis_ticks)
    ax.scatter(3,explained_variance_ratio_cumulated[3-1],c='blue',s=400)
    ax.grid(axis="x",linewidth=0.5)
    ax.grid(axis="y",linewidth=0.5)

# ### Most important features on EACH principal component (run on demand)

# +
# Get indices of the top 3 contributing columns / vars,
# i.e., the ones with the largest absolute values
important_3_cols = [np.argsort(np.abs(pca.components_[i]))[::-1][:3] for i in range(n_pcs)]
important_3_colnames = [list(acceptance_num.columns[important_3_cols[i]]) for i in range(n_pcs)]

importance_perc_3_cols = []
for i in range(n_pcs):
    percentages = [100*np.abs(pca.components_[i][important_3_cols[i][import_col_idx]])/np.sum(
        np.abs(pca.components_[i])
    ) for import_col_idx in range(3)]
    percentages = ["%.2f" % number+" %" for number in np.around(percentages,1)]
    
    importance_perc_3_cols.append(percentages)

# +
# PCs
PCs = pd.DataFrame(np.abs(pca.components_).round(2),
                 index=[f'PC{i+1}' for i in range(len(pca.components_))],
                 columns=acceptance_scaled.columns)

PCs["Important 3 columns"]=important_3_colnames
PCs["Importancies"]=importance_perc_3_cols
PCs.iloc[:,-2:]
# -

# ## K-Means clustering with 3 PCA columns

kmeans_pca = KMeans(n_clusters=4, random_state=42)
kmeans_pca.fit(acceptance_proj)
# Getting cluster numbers
labels_kmeans_pca = kmeans_pca.labels_
# Sorting clusters by size
# Get value counts as a 2d array
label_counts = np.asarray(np.unique(labels_kmeans_pca, return_counts=True)).T
# Sort label_counts in descending order by the second column
label_counts_sorted_desc = label_counts[label_counts[:, 1].argsort()[::-1]]
# Dictionary to set new cluster labels
label_convert_dict = dict(
    zip(label_counts_sorted_desc[:, 0], np.sort(label_counts_sorted_desc[:,
                                                                         0])))
# Translate labels according to the dictionary label_convert_dict
sorted_labels_kmeans_pca = np.vectorize(
    label_convert_dict.get)(labels_kmeans_pca)
# np.unique(sorted_labels_kmeans_pca, return_counts=True) # check-up new label counts

# Plotly visualization: A cluster plot that can be rotated etc
# As x,y,z we select uncorrelated variables
fig = px.scatter_3d(acceptance_proj,
                    x=acceptance_proj["pr_PC1"],
                    y=acceptance_proj["pr_PC2"],
                    z=acceptance_proj["pr_PC3"],
                    color=sorted_labels_kmeans_pca,
#                     color=np.char.mod('%d', sorted_labels_kmeans_pca), # to create a legend
                    width=500,
                    height=500,
                    title='3D Scatter plot for K-Means Clusters')
fig.update_traces(marker_size=6)  # reduce marker sizes
fig.update_layout(hovermode=False)  # remove hover info for x, y, z
fig.show()
fig.write_html("./output/clustering/kmeans/direct_with_pca/clusters_kmeans.html")  # interactive plot
fig.write_image(file="./output/clustering/kmeans/direct_with_pca/clusters_kmeans.jpg",
                format="jpg",scale=3)  # scale > 1 improves the resolution

# +
# matplotlib visualization of clusters (simple plot, run on demand)
cluster_col_dict = {0:"r", 1:"g", 2:"b", 3:"y"}
cluster_color_vec = np.vectorize(cluster_col_dict.get)(sorted_labels_kmeans_pca)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d')
ax.scatter(acceptance_proj["pr_PC1"],
           acceptance_proj["pr_PC2"],
           acceptance_proj["pr_PC3"],
           c = cluster_color_vec, marker='o',s=80)
plt.show()
# -

# ## K-Means clustering with specializations ("1.1") and 3 PCA columns (run on demand)

# We observe that including the categorical column  "1.1" in clustering yields bad results, see below:

acceptance_proj_ext = acceptance_proj.copy()
# Add the categorical column (unscaled)
acceptance_proj_ext["1.1"] = acceptance_imputed_df["1.1"]
# acceptance_proj_ext.head(3)

kmeans_pca_ext = KMeans(n_clusters = 4, random_state=42)
kmeans_pca_ext.fit(acceptance_proj_ext)
labels_kmeans_pca_ext = kmeans_pca_ext.labels_

# As x,y,z we select uncorrelated variables
fig = px.scatter_3d(acceptance_proj_ext, 
                    x = acceptance_proj_ext["pr_PC1"],
                    y = acceptance_proj_ext["pr_PC2"],
                    z = acceptance_proj_ext["pr_PC3"],
                    color = labels_kmeans_pca_ext, width=500, height=500)
fig.update_traces(marker_size = 6) # reduce marker sizes
fig.show()

# ## Performance Evaluations with YellowBricks (run on demand)

# 📚 We use a nice ***Data Visualisation*** library dedicated to Machine Learning algorithms which is called [**`YellowBricks`**].

# ### The Elbow Method

# Try to find the Elbow of the KMeans algorithm on `acceptance_cluster_df` using the ***KElbowVisualizer***. Note that KMeans is stochastic (the results may vary even if we run the function with the same inputs' values). Hence, we specify a value for the `random_state` parameter in order to make the results reproducible.

# +
# Instantiate the clustering model and visualizer
kmeans_model = KMeans(random_state=42)
elbow_visualizer = KElbowVisualizer(kmeans_model, k=(2,9))
# elbow_visualizer = KElbowVisualizer(kmeans, n_clusters = (2,8))

elbow_visualizer.fit(acceptance_proj) # Fit the data to the visualizer
elbow_plot = elbow_visualizer.poof() # or .show() # Finalize and render the figure
elbow_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"elbow_method_kmeans"+".pdf",
    format="pdf", bbox_inches="tight"
);
elbow_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"elbow_method_kmeans"+".jpg",
    format="jpg", bbox_inches="tight"
);
# -

# 👉 This `KElbowVisualizer` was able to detect the "elbow" at $K = 4$.

# ### The Silhouette Method

# +
range_n_clusters = list(range(2,9)) # [2, 3, 4, 5, 6, 7, 8]
num_clusters = len(range_n_clusters)
X = acceptance_proj.to_numpy() # to get rid of warnings

fig, ax = plt.subplots(num_clusters, 1,
                       figsize=(6,8),
                       constrained_layout=True,sharex=True)

for n_clusters in range_n_clusters:
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    clusterer.fit(X)
    cluster_labels = clusterer.labels_

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters = ",
        n_clusters,
        ", the average silhouette_score is: ",
        silhouette_avg,
        sep = ''
    )
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(clusterer, colors='yellowbrick',
                                      ax=ax[n_clusters-min(range_n_clusters)])
    visualizer.fit(X)

plt.savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"silhouette_scores_pca_kmean"+".pdf",
    format="pdf", bbox_inches="tight"
);
plt.savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"silhouette_scores_pca_kmean"+".jpg",
    format="jpg", bbox_inches="tight"
);
# https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
# -

# We note that for `n_clusters = 4` without PCA, the average silhouette_score was `0.1301`; now we have a score of `0.3242` which is much better!

# ## Properties of 4 K-Means clusters

# +
# # Another way of sorting clusters by size
# acceptance_imputed_df["cluster_no"] = labels_kmeans_pca
# kmeans_labels_by_size = list(acceptance_imputed_df["cluster_no"].value_counts().index)
# # Dictionary to set new cluster labels
# label_conv_dict = dict(zip(kmeans_labels_by_size,sorted(kmeans_labels_by_size)))
# # Label convertion
# acceptance_imputed_df['cluster_no'] = acceptance_imputed_df['cluster_no'].map(label_conv_dict)
# # Check clusters
# acceptance_imputed_df["cluster_no"].value_counts()

# +
# Getting cluster numbers
n_clusters = 4
acceptance_imputed_df["cluster_no"] = sorted_labels_kmeans_pca

# Subset data per cluster
acceptance_df_cluster_list = [
    acceptance_imputed_df[acceptance_imputed_df["cluster_no"] ==
                          i].iloc[:, :-1] for i in range(n_clusters)
]
# acceptance_df_cluster_list = [acceptance_imputed_df[
#     acceptance_imputed_df["cluster_no"]==i].iloc[:,:-1].iloc[:,4:] for i in range(5)]
# -

# Export clusters:
acceptance_imputed_df.iloc[:,-1:].to_csv('./data/clustered_data/pca_kmeans_clusters_direct.csv')

# +
# Another plotly visualization with sorted legend
# A cluster plot that can be rotated etc
# Prepare a sorted data frame to have a sorted legend
kmeans_plot_df = acceptance_proj.copy()
kmeans_plot_df["Group"] = 1+sorted_labels_kmeans_pca
kmeans_plot_df = kmeans_plot_df.sort_values('Group')
kmeans_plot_df["Group"] = kmeans_plot_df["Group"].astype("str")

# Plot and save
kmeans_fig_2 = px.scatter_3d(
    kmeans_plot_df,
    x=kmeans_plot_df["pr_PC1"],
    y=kmeans_plot_df["pr_PC2"],
    z=kmeans_plot_df["pr_PC3"],
    color='Group',
    width=500,
    height=500,
    #                            template='plotly_dark',
    title='3D Scatter plot for K-Means Clusters')
kmeans_fig_2.update_traces(marker_size=3.5)  # reduce marker sizes
kmeans_fig_2.update_layout(hovermode=False)  # remove hover info for x, y, z
kmeans_fig_2.show()
kmeans_fig_2.write_html("./output/clustering/kmeans/direct_with_pca/clusters_kmeans_v2.html")  # interactive plot
kmeans_fig_2.write_image(
    file="./output/clustering/kmeans/direct_with_pca/clusters_kmeans_v2.jpg",
    format="jpg", scale=3) # scale > 1 improves the resolution

# +
# A plotly visualization for the paper
# removed axis titles and the white space around

# Prepare a sorted data frame to have a sorted legend
kmeans_plot_df = acceptance_proj.copy()
kmeans_plot_df["Group"] = 1+sorted_labels_kmeans_pca
kmeans_plot_df = kmeans_plot_df.sort_values('Group')
kmeans_plot_df["Group"] = kmeans_plot_df["Group"].astype("str")

# Plot and save
kmeans_fig_3 = px.scatter_3d(
    data_frame = kmeans_plot_df,
    x=kmeans_plot_df["pr_PC1"],
    y=kmeans_plot_df["pr_PC2"],
    z=kmeans_plot_df["pr_PC3"],
    color='Group',
    width=500,
    height=450
#     title='3D Scatter plot for K-Means Clusters'
)
kmeans_fig_3.update_traces(marker_size=3.1)  # reduce marker sizes
kmeans_fig_3.update_layout(hovermode=False)  # remove hover info for x, y, z
kmeans_fig_3.layout['scene']['xaxis']['title']['text']="" # remove x label
kmeans_fig_3.layout['scene']['yaxis']['title']['text']="" # remove y label
kmeans_fig_3.layout['scene']['zaxis']['title']['text']="" # remove z label

# Set default zoom and remove white space
zoom_factor = 0.8
kmeans_fig_3.update_layout(
    legend_title="Clusters",
    legend=dict(
        orientation='h', # horizontal legend
        y=0.95,
        xanchor="center",
        x=0.5,
        itemwidth=35 # spacing
    ),
#     legend = dict(x=0.88,y=0.86), # alter legends placement
#     showlegend=False, # hide legend
#     scene = dict(
#         xaxis = dict(showticklabels = False),
#         yaxis = dict(showticklabels = False),
#         zaxis =dict(showticklabels=False)
#         ),
    scene_aspectratio=dict(x=zoom_factor,y=zoom_factor,z=zoom_factor),
    margin = {'l':0,'r':0,'t':0,'b':0}
)

kmeans_fig_3.update_traces(name="A", selector=dict(name="1"))
kmeans_fig_3.update_traces(name="B", selector=dict(name="2"))
kmeans_fig_3.update_traces(name="C", selector=dict(name="3"))
kmeans_fig_3.update_traces(name="D", selector=dict(name="4"))

kmeans_fig_3.show()
kmeans_fig_3.write_html("./output/clustering/kmeans/direct_with_pca/clusters_kmeans_v3.html")  # interactive plot
kmeans_fig_3.write_image(
    file="./output/clustering/kmeans/direct_with_pca/clusters_kmeans_v3.pdf",
    format="pdf", scale=2)
# -

# ### The averages and the standard deviations for the clusters

# +
xlab_angle = 55
kmeans_lineplot = acceptance_imputed_df.iloc[:, 1:-1].mean().plot(
    figsize=(10, 4),
    color="black",
    xticks=range(acceptance_imputed_df.iloc[:, 1:-1].shape[1]),
    fontsize = 9,
    rot=xlab_angle,
    label="Total ("+str(len(acceptance_imputed_df))+")"
)

for idx, df in enumerate(acceptance_df_cluster_list):
    df.iloc[:, 1:].mean().plot(
        label="Group " + str(idx + 1)+" ("+str(len(df))+")",
        rot=xlab_angle
    )

# Customizing
kmeans_lineplot.grid(linestyle='dotted', linewidth=0.4)
kmeans_lineplot.set_xlabel('Question code', size = 12, labelpad=15)
kmeans_lineplot.set_ylabel('Average grade', size = 12, labelpad=15)
kmeans_lineplot.legend(loc="lower left", bbox_to_anchor=(1, 0.6));
kmeans_lineplot.set_title("Averages per Group", size=14, pad=12)

kmeans_lineplot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"averages_groups_kmeans"+".jpg",
    format="jpg",bbox_inches="tight",dpi=200
)

# +
# 1-column dataframe for group averages for odd / even questions
group_averages_df_values = [acceptance_imputed_df.iloc[:,1:25:2].mean().mean(),
                           acceptance_imputed_df.iloc[:,2:25:2].mean().mean()]
group_averages_df_indices = ["Odds", "Evens"]
for idx, df in enumerate(acceptance_df_cluster_list):
    group_averages_df_values.append(df.iloc[:,1:25:2].mean().mean())
    group_averages_df_values.append(df.iloc[:,2:25:2].mean().mean())
    group_averages_df_indices.append("Odds, cluster "+str(idx+1))
    group_averages_df_indices.append("Evens, cluster "+str(idx+1))

group_averages_df = pd.DataFrame({"Average": group_averages_df_values},
                                 index =group_averages_df_indices)
# group_averages_df

# +
# 2-column dataframe for group averages for odd / even questions (desire/expectation)
# Column-vector for odds
average_odds_vec = [acceptance_imputed_df.iloc[:,1:25:2].mean().mean()] +\
[df.iloc[:,1:25:2].mean().mean() for idx, df in enumerate(acceptance_df_cluster_list)]
# Column-vector for evens
average_evens_vec = [acceptance_imputed_df.iloc[:,2:25:2].mean().mean()] +\
[df.iloc[:,2:25:2].mean().mean() for idx, df in enumerate(acceptance_df_cluster_list)]
# Indices
indices_vec = ["Total"] + ["Group "+str(idx+1) for idx in range(len(acceptance_df_cluster_list))]

# Create DataFrame
group_averages_2col_df = pd.DataFrame({"Average odds": average_odds_vec,
                                   "Average evens": average_evens_vec},
                                  index = indices_vec)
group_averages_2col_df

# +
# Visualizing group averages for odd / even questions compared to the total ones
fig_group_avg_odd_even = group_averages_2col_df.plot.barh(
    figsize=(9, 3),
    width=0.6,
#     xlim=(0, 0.5),
#     ylim=(-2, 6),
    edgecolor='white',
    color=["Crimson","SlateBlue"], # ["LightCoral","cadetblue","green"]
    linewidth=2,
    rot=20
)

# Customizing
fig_group_avg_odd_even.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_group_avg_odd_even.set_xlabel('Average Desire / Expectation', labelpad=12)
fig_group_avg_odd_even.set_ylabel('Student Groups', labelpad=20)

# # Sort legend titles by name and alter its placement
fig_group_avg_odd_even.legend(
    labels = ["Desire","Expectation"],
    loc="center left",
    bbox_to_anchor = (1, 0.5)
)

fig_group_avg_odd_even.get_figure().gca().invert_yaxis() # To invert the vertical axis

fig_group_avg_odd_even.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"group_avg_odd_even"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_group_avg_odd_even.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"group_avg_odd_even"+".pdf",
    format="pdf", bbox_inches="tight"
);

# +
# 2-column comparision dataframe for group averages for odd / even questions
# We subtract the total averages from the group ones
averages_compar_df= group_averages_2col_df.copy()
for row_idx in range(1,group_averages_2col_df.shape[0]):
    averages_compar_df.iloc[row_idx,:] = averages_compar_df.iloc[row_idx,:]-averages_compar_df.iloc[0,:]
        
averages_compar_df = averages_compar_df.iloc[1:,:]
averages_compar_df

# +
# Visualizing group averages for odd / even questions compared to the total ones
fig_group_avg_compare = averages_compar_df.plot.barh(
    figsize=(9, 3),
    width=0.6,
    xlim=(-1.6, 1),
#     ylim=(-2, 6),
    edgecolor='white',
    color=["Crimson","SlateBlue"], # ["LightCoral","cadetblue","green"]
    linewidth=1,
    rot=20
)

# Customizing
fig_group_avg_compare.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_group_avg_compare.set_xlabel('Desire / Expectation (compared with the total ones)', labelpad=12)
fig_group_avg_compare.set_ylabel('Student Groups', labelpad=25)

# # Sort legend titles by name and alter its placement
fig_group_avg_compare.legend(
    labels = ["Desire","Expectation"],
    loc="center left",
    bbox_to_anchor = (1, 0.5)
);

fig_group_avg_compare.get_figure().gca().invert_yaxis() # To invert the vertical axis

fig_group_avg_compare.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"group_avg_compare"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_group_avg_compare.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"group_avg_compare"+".pdf",
    format="pdf", bbox_inches="tight"
);

# +
# Plot for standard deviations
acceptance_imputed_df.iloc[:,1:-1].std().plot(
    figsize=(10,4),
    color="black",
    xticks=range(acceptance_imputed_df.iloc[:, 1:-1].shape[1]), # show all x-values
    fontsize = 9,
    title="Standard deviations",
    label="Total"
)

for idx, df in enumerate(acceptance_df_cluster_list):
    df.iloc[:,1:].std().plot(label="Group "+str(idx+1),rot=48)

plt.grid(linestyle='dotted', linewidth=0.4)
plt.legend(loc="lower left", bbox_to_anchor = (1, 0.6));
plt.savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"deviations_groups"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
);
# -

# > Thus, the clustering does not change the averages or the standard deviations of the grades. We can still observe grade deviations inside each of the groups.

# ### Important Question Types and Clusters

# Based on the line plot "Averages per Group" (see above), we can call the clustered student groups as follows:

# +
cluster_names = ["Enthusiasts","Realists", "Cautious","Indifferent"]
cluster_sizes = np.array([len(df) for df in acceptance_df_cluster_list])

# Add cluster names and sizes to the index:
means_groups_idx_vec = cluster_names
means_groups_idx_vec = np.char.add(means_groups_idx_vec,np.repeat(" (",n_clusters))
means_groups_idx_vec = np.char.add(means_groups_idx_vec,cluster_sizes.astype(str))
means_groups_idx_vec = np.char.add(means_groups_idx_vec,np.repeat(")",n_clusters))
means_groups_index = ["All together ("+str(len(acceptance_imputed_df))+")"]+list(means_groups_idx_vec)
# -

# We compute average scores per question type for each group:

# Calculate mean value within each cluster
means_list = [row_mean_features_df.mean()]
for idx in range(4):
    means_list.append(row_mean_features_df.loc[acceptance_imputed_df["cluster_no"] == idx].mean())
means_group_df = pd.concat(means_list, axis=1).T
means_group_df=means_group_df.set_index(pd.Index(means_groups_index))
means_group_df.to_html("./output/clustering/kmeans/direct_with_pca/"+"means_groups_Kmeans_dir.html")
means_group_df

# +
# # The plot for DM Meeting slides on 06.04.2023
# fig_desire_expec_avg_group = means_group_df.iloc[:,0:4].plot.bar(
#     figsize=(8.5, 2.5),
#     width=0.65,
#     rot=0,
#     linewidth=2,
#     color=['#4F81BD', '#C0504D',"#00CDCD","#8B008B"], # matched with presentation colors
#     edgecolor='white'
# )

# # Customizing
# fig_desire_expec_avg_group.grid(axis="y", linestyle='dotted', linewidth=0.35)
# fig_desire_expec_avg_group.set_xlabel('Student groups', size="medium", labelpad=14)
# fig_desire_expec_avg_group.set_ylabel('Average grades', size = "medium", labelpad=20)
# fig_desire_expec_avg_group.tick_params(axis='x', labelsize=9.5)

# # Alter legend's placement
# fig_desire_expec_avg_group.legend(
#     loc="center left",
#     fontsize = "x-small", # 'xx-small', 'x-small', 'small', 'medium'
#     bbox_to_anchor = (1, 0.7)
# )

# fig_desire_expec_avg_group.get_figure().savefig(
#     "./output/clustering/kmeans/direct_with_pca/"+"desire_expec_avg_group_brief"+".jpg",
#     format="jpg", bbox_inches="tight", dpi = 200
# )
# fig_desire_expec_avg_group.get_figure().savefig(
#     "./output/clustering/kmeans/direct_with_pca/"+"desire_expec_avg_group_brief"+".pdf",
#     format="pdf", bbox_inches="tight"
# );

# +
# The plot for LaTeX paper
fig_desire_expec_avg_group = means_group_df.iloc[:,0:4].plot.bar(
    figsize=(8.5, 2.5),
    width=0.65,
    rot=0,
    linewidth=2,
    ylim=(0, 8),
#     color=['#4F81BD', '#C0504D',"#00CDCD","#8B008B"], # matched with presentation colors
    color=['#4F81BD', '#C0504D',"cadetblue","darksalmon"], #  "tan","darksalmon","indianred","darkgoldenrod","peru","tab:brown"
    edgecolor='white'
)

# Customizing
fig_desire_expec_avg_group.xaxis.grid()
fig_desire_expec_avg_group.set_yticks(np.arange(0, 7, 2))
fig_desire_expec_avg_group.grid(axis="y", linestyle='dashed', linewidth=0.4)
# fig_desire_expec_avg_group.set_xlabel('Student groups', size="medium", labelpad=14)
fig_desire_expec_avg_group.set_ylabel('Average grades', size = 13.5, labelpad=15)
fig_desire_expec_avg_group.tick_params(axis='x', labelsize=10.5)

# Alter legend's placement
fig_desire_expec_avg_group.legend(
    loc="center",
    ncol=4,
    fontsize = 11.6, # 'xx-small', 'x-small', 'small', 'medium', large, x-large, xx-large, larger, smaller
    bbox_to_anchor = (0.5, 0.92)
)

fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"desire_expec_avg_group_brief"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"desire_expec_avg_group_brief"+".pdf",
    format="pdf", bbox_inches="tight"
);

# +
fig_desire_expec_avg_group = means_group_df.drop(means_group_df.columns[[2,3]],axis = 1).plot.bar(
    figsize=(8.5, 2.5),
    width=0.65,
    rot=0,
    linewidth=2,
    color=['#4F81BD', '#C0504D',"SlateBlue","Crimson","cadetblue","LightCoral"],
    edgecolor='white'
)

# Customizing
fig_desire_expec_avg_group.grid(axis="y", linestyle='dotted', linewidth=0.35)
fig_desire_expec_avg_group.set_xlabel('Student groups', size="medium", labelpad=14)
fig_desire_expec_avg_group.set_ylabel('Average grades', size = "medium", labelpad=20)
fig_desire_expec_avg_group.tick_params(axis='x', labelsize="small")

# Alter legend's placement
fig_desire_expec_avg_group.legend(
    loc="center left",
    fontsize = "x-small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (1, 0.7)
)

fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"desire_expec_avg_group_detailed"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"desire_expec_avg_group_detailed"+".pdf",
    format="pdf", bbox_inches="tight"
);
# -

# ### Specialization distribution dataframe per cluster

# +
# Specialization distribution dataframe per cluster
# Get value counts as 2d arrays, then calculate the percentage for each cluster
spec_totals = np.asarray(np.unique(acceptance_imputed_df["1.1"], return_counts=True))[1,:]

spec_counts_list = [
    np.asarray(np.unique(
        acceptance_df_cluster_list[cluster_id]["1.1"], return_counts=True
    ))[1,:] for cluster_id in range(n_clusters)
]

spec_percentages_list = [
    100*spec_counts_list[cluster_id]/spec_totals for cluster_id in range(n_clusters)
]

spec_percentages_df=pd.concat([pd.Series(x) for x in spec_percentages_list], axis=1)
spec_percentages_df.index=specializations_en
spec_percentages_df.columns=cluster_names
spec_percentages_df = spec_percentages_df.sort_values(by=["Enthusiasts"])
spec_percentages_df

# +
# Bar chart for specialization distributions per cluster
spec_distrib_cluster_bplot = spec_percentages_df.plot.barh(
    color=["tab:blue","peru",'tab:cyan', '#C0504D'],
    edgecolor='white',
    width=0.75
)

# Customizing
spec_distrib_cluster_bplot.invert_yaxis()
spec_distrib_cluster_bplot.grid(axis="y", linestyle='dotted', linewidth=0.05)
spec_distrib_cluster_bplot.set_xlabel('Percentage', size="large", labelpad=14)
spec_distrib_cluster_bplot.tick_params(axis='x', labelsize="medium")
spec_distrib_cluster_bplot.tick_params(axis='y', labelsize=14)

# Alter legend's placement
spec_distrib_cluster_bplot.legend(
    loc="upper center",
    fontsize = "medium", # xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller
    ncol=4,
    bbox_to_anchor = (0.5, 1.1)
)

spec_distrib_cluster_bplot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"specialization_dist_bar"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
spec_distrib_cluster_bplot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"specialization_dist_bar"+".pdf",
    format="pdf", bbox_inches="tight"
);

# +
# Bar chart for specialization distributions per cluster
spec_distrib_cluster_plot = spec_percentages_df.plot.barh(
    color=["tab:blue","peru",'tab:cyan', '#C0504D'],
    stacked=True,
    edgecolor='white',
    width=0.7,
)
spec_distrib_cluster_plot.invert_yaxis()

# Customizing
spec_distrib_cluster_plot.grid(axis="y", linestyle='dotted', linewidth=0.05)
spec_distrib_cluster_plot.set_xlabel('Percentage', size="large", labelpad=14)
spec_distrib_cluster_plot.tick_params(axis='x', labelsize="medium")
spec_distrib_cluster_plot.tick_params(axis='y', labelsize="large")

# Alter legend's placement
spec_distrib_cluster_plot.legend(
    loc="upper center",
    fontsize = "medium", # xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller
    ncol=4,
    bbox_to_anchor = (0.5, 1.1)
)

spec_distrib_cluster_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"specialization_dist"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
spec_distrib_cluster_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"specialization_dist"+".pdf",
    format="pdf", bbox_inches="tight"
);
# -

# Since we have here 3 small student groups, some results are not significant. Hence we will merge the groups under 30 members and perform the visualization for the merged specializations.

# +
# Merge "Electrical engineering", "Mechanical engineering" and "Mechatronics"
spec_totals_merged = np.copy(spec_totals)
spec_totals_merged[2] = spec_totals_merged[2] +spec_totals_merged[4] +spec_totals_merged[5]
spec_totals_merged = np.delete(spec_totals_merged,[4,5])


spec_counts_merged_list = spec_counts_list.copy()
for cluster_id in range(n_clusters):
    spec_counts_merged_list[cluster_id][2] = spec_counts_merged_list[cluster_id][2] + \
    spec_counts_merged_list[cluster_id][4] + spec_counts_merged_list[cluster_id][5]
    spec_counts_merged_list[cluster_id] = np.delete(spec_counts_merged_list[cluster_id],[4,5])   

# +
spec_merged_percentages_list = [
    100*spec_counts_merged_list[cluster_id]/spec_totals_merged for cluster_id in range(n_clusters)
]

spec_merged_percentages_df = pd.concat([pd.Series(x) for x in spec_merged_percentages_list], axis=1)
spec_merged_percentages_df.index = specializations_merged
spec_merged_percentages_df.columns = cluster_names
spec_merged_percentages_df = spec_merged_percentages_df.sort_values(by=["Enthusiasts"])
spec_merged_percentages_df

# +
# Bar chart for merged specialization distributions per cluster
spec_merged_distrib_cluster_plot = spec_merged_percentages_df.plot.barh(
    color=["tab:blue","peru",'tab:cyan', '#C0504D'],
    stacked=True,
    edgecolor='white',
    width=0.7,
)
spec_merged_distrib_cluster_plot.invert_yaxis()

# Customizing
spec_merged_distrib_cluster_plot.grid(axis="y", linestyle='dotted', linewidth=0.05)
spec_merged_distrib_cluster_plot.set_xlabel('Percentage', size="large", labelpad=14)
spec_merged_distrib_cluster_plot.tick_params(axis='x', labelsize="medium")
spec_merged_distrib_cluster_plot.tick_params(axis='y', labelsize="large")

# Alter legend's placement
spec_merged_distrib_cluster_plot.legend(
    loc="upper center",
    fontsize = "medium", # xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller
    ncol=4,
    bbox_to_anchor = (0.5, 1.1)
)

spec_merged_distrib_cluster_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"specialization_merged_dist"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
spec_merged_distrib_cluster_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_with_pca/"+"specialization_merged_dist"+".pdf",
    format="pdf", bbox_inches="tight"
);
# -

# ### Plot histograms per cluster (run on demand)

# +
# %%time
# takes 23 sec

for idx, df in enumerate(acceptance_df_cluster_list):
    suptitle = f"K-Means: Group {idx+1} of {n_clusters}, {len(df)} students of {len(acceptance_imputed_df)}"
    plot_df_histograms(
        df,
        output_name="clustering/kmeans/direct_with_pca/histograms_cluster_"+str(idx+1),
        suptitle = suptitle
    )
# -

# By comparing the plots for 1.1 corresponding to each group, we see that the distribution is roughly the same. I.e., we cannot link the members of a particular cluster to a particular specialization.
