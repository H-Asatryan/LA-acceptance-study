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

# # Acceptance Study: K-Means Clustering with Mean Features

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
from functions_and_variables import plot_df_histograms, specializations

# +
# Load data
row_mean_features_df = pd.read_csv("./data/clean_data/row_mean_features.csv",index_col="SurveyID")

# Drop the general LA columns and keep the partitioned LA ones
# (otherwise we have strong correlations etc)
row_mean_features_df = row_mean_features_df.drop(columns=["LA Desire", "LA Expectation"])
row_mean_features_df.head(2)
# -

# ## Scaling

# Scaling numerical features for PCA / K-Means
scaler = RobustScaler()
row_mean_features_scaled = pd.DataFrame(
    scaler.fit_transform(row_mean_features_df),
    columns=row_mean_features_df.columns)

# ### Heatmaps for the scaled data (run on demand)

# +
corr_matrix = row_mean_features_scaled.corr()

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
    "./output/docs/dataset_info/"+"heatmap_scaled_avg_feat"+".pdf",
    format="pdf", bbox_inches="tight");
plt.savefig(
    "./output/docs/dataset_info/"+"heatmap_scaled_avg_feat"+".jpg",
    format="jpg", bbox_inches="tight");
# -

plt.figure(figsize=(16,10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", mask=mask,
    xticklabels=corr_matrix.columns,
    yticklabels=corr_matrix.columns,cmap= "bwr"
);
plt.savefig(
    "./output/docs/dataset_info/"+"heatmap_scaled_avg_feat_above_0_3"+".pdf",
    format="pdf", bbox_inches="tight");
plt.savefig(
    "./output/docs/dataset_info/"+"heatmap_scaled_avg_feat_above_0_3"+".jpg",
    format="jpg", bbox_inches="tight");

# After comparing these correlations to those without scaling / imputing we observe that the values here are slightly smaller that the old ones. For the largest values we have a difference of about `0.01`.

# ## PCA (run on demand)

# ### PCA initialization

pca = PCA()
pca.fit(row_mean_features_scaled); # find all the principal components (no target!)

# Now we project our `row_mean_features_scaled` dataset onto the new space with the number of principal components we decided to keep. We name it `row_mean_features_proj`

# +
n_pcs = 3 # Threshold PCA
pca = PCA(n_components=n_pcs, whiten=True)
pca.fit(row_mean_features_scaled)
row_mean_features_proj = pd.DataFrame(pca.transform(row_mean_features_scaled))
# acceptance_proj.head(2)

# This command also prevents NaN values after column assignment
row_mean_features_proj = row_mean_features_proj.set_index(row_mean_features_df.index)
# Naming PCA features for plotly visualizations
row_mean_features_proj.columns = ["pr_PC"+str(i+1) for i in range(n_pcs)]
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
important_3_colnames = [list(row_mean_features_df.columns[important_3_cols[i]]) for i in range(n_pcs)]

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
                 columns=row_mean_features_scaled.columns)

PCs["Important 3 columns"]=important_3_colnames
PCs["Importancies"]=importance_perc_3_cols
PCs.iloc[:,-2:]
# -

# We do not observe significant advantages to use PCA; we will keep all 6 variables and perform direct clustering. This way we are going to have an explainable algorithm.

# ## K-Means clustering without PCA

n_clusters = 4
kmeans_row_means = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_row_means.fit(row_mean_features_scaled)
# Getting cluster numbers
labels_kmeans_row_means = kmeans_row_means.labels_
# Sorting clusters by size
# Get value counts as a 2d array
label_counts = np.asarray(np.unique(labels_kmeans_row_means, return_counts=True)).T
# Sort label_counts in descending order by the second column
label_counts_sorted_desc = label_counts[label_counts[:, 1].argsort()[::-1]]
# Dictionary to set new cluster labels
label_convert_dict = dict(
    zip(label_counts_sorted_desc[:, 0],
        np.sort(label_counts_sorted_desc[:,0]))
)
# Translate labels according to the dictionary label_convert_dict
sorted_labels_kmeans_row_means = np.vectorize(
    label_convert_dict.get)(labels_kmeans_row_means)
# np.unique(sorted_labels_kmeans_pca, return_counts=True) # check-up new label counts

# ## Performance Evaluations with YellowBricks (run on demand)

# 📚 We use a nice ***Data Visualisation*** library dedicated to Machine Learning algorithms which is called [**`YellowBricks`**].

# ### The Elbow Method

# Try to find the Elbow of the KMeans algorithm on `acceptance_cluster_df` using the ***KElbowVisualizer***. Note that KMeans is stochastic (the results may vary even if we run the function with the same inputs' values). Hence, we specify a value for the `random_state` parameter in order to make the results reproducible.

# +
# Instantiate the clustering model and visualizer
kmeans_model = KMeans(random_state=42)
elbow_visualizer = KElbowVisualizer(kmeans_model, k=(2,9))
# elbow_visualizer = KElbowVisualizer(kmeans, n_clusters = (2,8))

elbow_visualizer.fit(row_mean_features_scaled) # Fit the data to the visualizer
elbow_plot = elbow_visualizer.poof() # or .show() # Finalize and render the figure
elbow_plot.get_figure().savefig(
    "./output/clustering/kmeans/via_row_avg_ft/"+"elbow_method_kmean"+".pdf",
    format="pdf", bbox_inches="tight");
elbow_plot.get_figure().savefig(
    "./output/clustering/kmeans/via_row_avg_ft/"+"elbow_method_kmean"+".jpg",
    format="jpg", bbox_inches="tight");
# -

# 👉 This `KElbowVisualizer` was able to detect the "elbow" at $K = 4$.

# ### The Silhouette Method

# +
range_n_clusters = list(range(2,9)) # [2, 3, 4, 5, 6, 7, 8]
num_clusters = len(range_n_clusters)
X = row_mean_features_scaled.to_numpy() # to get rid of warnings

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
    "./output/clustering/kmeans/via_row_avg_ft/"+"silhouette_scores_kmean"+".pdf",
    format="pdf", bbox_inches="tight");
plt.savefig(
    "./output/clustering/kmeans/via_row_avg_ft/"+"silhouette_scores_kmean"+".jpg",
    format="jpg", bbox_inches="tight");
# https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
# -

# Therefore we choose `n_clusters = 4` without PCA (though `n_clusters = 5` has a better silhouette score, it has longer tails and hence more outliers).

# ## Properties of 4 K-Means clusters

# +
# Getting cluster numbers
n_clusters=4
row_mean_features_df["cluster_no"] = sorted_labels_kmeans_row_means

# Subset data per cluster
row_mean_features_cluster_list = [row_mean_features_df[
    row_mean_features_df["cluster_no"]==i].iloc[:,:-1] for i in range(n_clusters)]

# Compute cluster sizes
row_mean_features_cluster_sizes = np.array([len(df) for df in row_mean_features_cluster_list])

# Export clusters
row_mean_features_df.iloc[:,-1:].to_csv('./data/clustered_data/row_avg_kmeans_clusters.csv')
# -

# We compute average scores per question type for each group:

# +
# Calculate feature averages in total / within each cluster
# and save them to a list
mean_feat_total_avg = row_mean_features_df.iloc[:, :-1].mean()
means_list = [mean_feat_total_avg]

for idx in range(4):
    means_list.append(row_mean_features_df.loc[row_mean_features_df["cluster_no"] == idx].iloc[:,:-1].mean())
means_group_df = pd.concat(means_list, axis=1).T

# Create a detailed list index, add group IDs and sizes to the index
# Uncomment one of the following lines to display clusters by their number or description,
# respectively (to reveal the descriptions we refer to the discussion after plots)
# cluster_names = np.char.add(np.repeat("Group ",n_clusters),np.arange(1,5).astype(str))
cluster_names = ["Enthusiasts","Realists", "Cautious","Indifferent"]

means_groups_idx_vec = cluster_names
means_groups_idx_vec = np.char.add(means_groups_idx_vec,np.repeat(" (",n_clusters))
means_groups_idx_vec = np.char.add(means_groups_idx_vec,row_mean_features_cluster_sizes.astype(str))
means_groups_idx_vec = np.char.add(means_groups_idx_vec,np.repeat(")",n_clusters))
means_groups_index = ["All together ("+str(len(row_mean_features_df))+")"]+list(means_groups_idx_vec)
means_group_df=means_group_df.set_index(pd.Index(means_groups_index))
means_group_df

# +
# Bar chart for feature averages in total / within each cluster
fig_desire_expec_avg_group = means_group_df.iloc[:,0:6].plot.bar(
    figsize=(8, 2.5),
    width=0.65,
    rot=0,
    linewidth=2,
    color=['#4F81BD', '#C0504D',"cadetblue","LightCoral","#00CDCD","#8B008B"],
    edgecolor='white'
)

# Customizing
fig_desire_expec_avg_group.grid(axis="x", linestyle='dotted', linewidth=0.3)
fig_desire_expec_avg_group.set_xlabel('Student groups', size="medium", labelpad=14)
fig_desire_expec_avg_group.set_ylabel('Average grades', size = "medium", labelpad=20)
fig_desire_expec_avg_group.tick_params(axis='x', labelsize="small")
fig_desire_expec_avg_group.set_title("Grade Averages per Group (K-Means Clustering)", size=12, pad=12)

# Alter legend's placement
fig_desire_expec_avg_group.legend(
    loc="center left",
    fontsize = "x-small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (1, 0.7)
)

fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/kmeans/via_row_avg_ft/"+"desire_exp_avg_row_feat"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/kmeans/via_row_avg_ft/"+"desire_exp_avg_row_feat"+".pdf",
    format="pdf", bbox_inches="tight"
);
# -

# ### The averages and the standard deviations for the clusters

# +
xlab_angle = 0

kmeans_lineplot = means_group_df.T.plot(
    figsize=(10, 4),
    linewidth=1.5,
    color=["black",'#4F81BD',"green", '#C0504D',"#8B008B"],
    fontsize = 8.5,
    rot=xlab_angle
)
    
# Customizing
kmeans_lineplot.grid(axis="both",linewidth=0.4)
kmeans_lineplot.set_xlabel('Question Group', size = 12, labelpad=15)
kmeans_lineplot.set_ylabel('Average Grade', size = 12, labelpad=15)
kmeans_lineplot.legend(loc="lower left", bbox_to_anchor=(1, 0.6));
kmeans_lineplot.set_title("Grade Averages per Group", size=14, pad=12)

kmeans_lineplot.get_figure().savefig(
    "./output/clustering/kmeans/via_row_avg_ft/"+"row_avg_groups_kmeans"+".jpg",
    format="jpg", bbox_inches="tight", dpi=200)
# -
# Based on the line plot "Averages per Group" (see above), we can call the clustered student groups as follows:
# *"Enthusiasts","Realists", "Cautious","Indifferent".*

# ## Intersection of two clustering approaches

# We recall that our direct clustering approach in "07_PCA_and_Kmeans.py" uses all the survey sheets (494); we have imputed the missing data there. The clustering by means of the row mean features uses a slightly reduced set (consisting of 463 sheets), since we drop some NAs. Hence we will first merge the cluster number data frames and remove NAs. Then we will compute the intersection of any direct cluster with any row means cluster. The resulting matrix with give a clue about the equality of the generated clusters.

# +
# Load direct PCA+K-Means cluster numbers:
pca_kmeans_clusters_direct_df = pd.read_csv(
    "./data/clustered_data/pca_kmeans_clusters_direct.csv",index_col="SurveyID")
pca_kmeans_clusters_direct_df = pca_kmeans_clusters_direct_df.rename(
    columns = {'cluster_no':'direct_cl_no'})

# Merge with row mean feature based cluster numbers and drop NAs
cluster_num_merged = pca_kmeans_clusters_direct_df.join(row_mean_features_df.iloc[:,-1:]).rename(
    columns={"cluster_no":"row_means_cl_no"}).dropna().astype({'row_means_cl_no':'int'})
cluster_num_merged.head(3)

# +
cluster_pivot = np.zeros((n_clusters,n_clusters),dtype=int)

for dir_cl_idx in range(n_clusters):
    for row_avg_cl_idx in range(n_clusters):
        cluster_pivot[dir_cl_idx,row_avg_cl_idx] = len(np.intersect1d(
            np.array(cluster_num_merged[cluster_num_merged["direct_cl_no"]==dir_cl_idx].index),
            np.array(cluster_num_merged[cluster_num_merged["row_means_cl_no"]==row_avg_cl_idx].index)
            ))

cluster_pivot_df = pd.DataFrame.from_records(cluster_pivot)
cluster_pivot_df.columns = np.char.add(np.repeat("row_avg_cl_",4),np.arange(1,n_clusters+1).astype(str))
cluster_pivot_df.index = np.char.add(np.repeat("direct_cl_",4),np.arange(1,n_clusters+1).astype(str))
cluster_pivot_df.to_html("./output/clustering/kmeans/kmeans_comparison_pivot_2_approaches.html")
cluster_pivot_df
