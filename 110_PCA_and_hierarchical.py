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

# # Acceptance Study: PCA and Hierarchical Clustering

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
from sklearn.cluster import AgglomerativeClustering
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

# ## Data scaling

# Scaling numerical features for PCA
scaler = RobustScaler()
row_mean_features_scaled = pd.DataFrame(scaler.fit_transform(row_mean_features_df),
                                        columns=row_mean_features_df.columns)

# ## Agglomerative (hierarchical) clustering without PCA

# +
n_clusters = 4
hierar_clusterer = AgglomerativeClustering(
#     linkage='complete', # default='ward', 'complete', 'average', 'single' 
    n_clusters=n_clusters
) # There is no random seed here, the default ward is the best option!

hierar_clusterer.fit(row_mean_features_scaled)

# Getting cluster numbers
labels_hierar = hierar_clusterer.labels_

# Sorting clusters by size
# Get value counts as a 2d array
label_counts = np.asarray(np.unique(labels_hierar, return_counts=True)).T
# Sort label_counts in descending order by the second column
label_counts_sorted_desc = label_counts[label_counts[:, 1].argsort()[::-1]]
# Dictionary to set new cluster labels
label_convert_dict = dict(
    zip(label_counts_sorted_desc[:, 0],
        np.sort(label_counts_sorted_desc[:,0])))
# Translate labels according to the dictionary label_convert_dict
sorted_labels_hierar = np.vectorize(
    label_convert_dict.get)(labels_hierar)
# np.unique(sorted_labels_hierar, return_counts=True) # check-up new label counts
# -

# ### Properties of 4 Ward Clusters

# +
# Getting cluster numbers
n_clusters=4
row_mean_features_df["cluster_no"] = sorted_labels_hierar

# Subset data per cluster
row_mean_features_cluster_list = [row_mean_features_df[
    row_mean_features_df["cluster_no"]==i].iloc[:,:-1] for i in range(n_clusters)]

# Compute cluster sizes
row_mean_features_cluster_sizes = np.array([len(df) for df in row_mean_features_cluster_list])

# Export clusters
row_mean_features_df.iloc[:,-1:].to_csv('./data/clustered_data/row_avg_ward_clusters.csv')

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
cluster_names = ["Realists","Enthusiasts","Indifferent","Cautious"]

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
fig_desire_expec_avg_group.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_desire_expec_avg_group.set_xlabel('Student groups', size="medium", labelpad=14)
fig_desire_expec_avg_group.set_ylabel('Average grades', size = "medium", labelpad=20)
fig_desire_expec_avg_group.tick_params(axis='x', labelsize="small")
fig_desire_expec_avg_group.set_title("Grade Averages per Group (Ward Clustering)",
                                     size=12, pad=12)

# Alter legend's placement
fig_desire_expec_avg_group.legend(
    loc="center left",
    fontsize = "x-small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (1, 0.7)
)

fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/ward/"+"desire_exp_avg_row_ft_ward"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/ward/"+"desire_exp_avg_row_ft_ward"+".pdf",
    format="pdf", bbox_inches="tight"
);
# -

# ### Intersection of K-Means and Ward clustering approaches with row mean features

# +
# Load direct PCA+K-Means cluster numbers:
row_mean_kmeans_clusters_df = pd.read_csv(
    "./data/clustered_data/row_avg_kmeans_clusters.csv",index_col="SurveyID")
row_mean_kmeans_clusters_df = row_mean_kmeans_clusters_df.rename(
    columns = {'cluster_no':'kmeans_cl_no'})

# Merge with row mean feature based cluster numbers and drop NAs
cluster_num_merged = row_mean_kmeans_clusters_df.join(row_mean_features_df.iloc[:,-1:]).rename(
    columns={"cluster_no":"ward_cl_no"}).dropna().astype({'ward_cl_no':'int'})
cluster_num_merged.head(3)

# +
cluster_pivot = np.zeros((n_clusters,n_clusters),dtype=int)

for dir_cl_idx in range(n_clusters):
    for row_avg_cl_idx in range(n_clusters):
        cluster_pivot[dir_cl_idx,row_avg_cl_idx] = len(np.intersect1d(
            np.array(cluster_num_merged[cluster_num_merged["kmeans_cl_no"]==dir_cl_idx].index),
            np.array(cluster_num_merged[cluster_num_merged["ward_cl_no"]==row_avg_cl_idx].index)
            ))

cluster_pivot_df = pd.DataFrame.from_records(cluster_pivot)
cluster_pivot_df.columns = np.char.add(np.repeat("kmeans_cl_",4),
                                       np.arange(1,n_clusters+1).astype(str))
cluster_pivot_df.index = np.char.add(np.repeat("ward_cl_",4),
                                     np.arange(1,n_clusters+1).astype(str))
cluster_pivot_df.to_html("./output/clustering/ward/comparison_with_kmeans_pivot.html")
cluster_pivot_df
# -
# We see that only the group "Indifferents" is almost equally structured in both clustering approaches. The other 3 groups of ward clustering are worse interpretable than the K-Means ones!


# ## PCA

# ### PCA initialization

pca = PCA()
pca.fit(row_mean_features_scaled); # find all the principal components (no target!)

# Now we project our `row_mean_features_scaled` dataset onto the new space with the number of principal components we decided to keep. We name it `row_mean_features_proj`:

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

# ## Agglomerative (hierarchical) clustering with 3 PCA columns

# +
n_clusters = 4
hierar_pca = AgglomerativeClustering(
#     linkage='complete', # default='ward', 'complete', 'average', 'single' 
    n_clusters=n_clusters
) # There is no random seed here, the default ward is the best option!

hierar_pca.fit(row_mean_features_proj)

# Getting cluster numbers
labels_hierar_pca = hierar_pca.labels_

# Sorting clusters by size
# Get value counts as a 2d array
label_counts = np.asarray(np.unique(labels_hierar_pca, return_counts=True)).T
# Sort label_counts in descending order by the second column
label_counts_sorted_desc = label_counts[label_counts[:, 1].argsort()[::-1]]
# Dictionary to set new cluster labels
label_convert_dict = dict(
    zip(label_counts_sorted_desc[:, 0],
        np.sort(label_counts_sorted_desc[:,0])))
# Translate labels according to the dictionary label_convert_dict
sorted_labels_hierar_pca = np.vectorize(
    label_convert_dict.get)(labels_hierar_pca)
# np.unique(sorted_labels_hierar_pca, return_counts=True) # check-up new label counts
# -

# Plotly visualization: A cluster plot that can be rotated etc
# As x,y,z we select uncorrelated variables
fig = px.scatter_3d(row_mean_features_proj,
                    x=row_mean_features_proj["pr_PC1"],
                    y=row_mean_features_proj["pr_PC2"],
                    z=row_mean_features_proj["pr_PC3"],
                    color=sorted_labels_hierar_pca,
#                     color=np.char.mod('%d', sorted_labels_hierar_pca), # to create a legend
                    width=500,
                    height=500,
                    title='3D Scatter plot for Hierarchical Clusters (Ward)')
fig.update_traces(marker_size=4)  # reduce marker sizes
fig.update_layout(hovermode=False)  # remove hover info for x, y, z
fig.show()
fig.write_html("./output/clustering/ward/clusters_pca_ward.html")  # interactive plot
fig.write_image(file="./output/clustering/ward/clusters_pca_ward.jpg", format="jpg",
                scale=3)  # scale > 1 improves the resolution

# ### Properties of 4 Ward Clusters with PCA

# +
# Getting cluster numbers
n_clusters=4
row_mean_features_df["cluster_no"] = sorted_labels_hierar_pca

# Subset data per cluster
row_mean_features_cluster_list = [row_mean_features_df[
    row_mean_features_df["cluster_no"]==i].iloc[:,:-1] for i in range(n_clusters)]

# Compute cluster sizes
row_mean_features_cluster_sizes = np.array([len(df) for df in row_mean_features_cluster_list])

# Export clusters
row_mean_features_df.iloc[:,-1:].to_csv('./data/clustered_data/row_avg_pca_ward_clusters.csv')
# -

# #### Average Scores per Group and Question

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
cluster_names = ["Enthusiasts","Indifferent","Realists", "Cautious"]

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
fig_desire_expec_avg_group.grid(axis="y", linestyle='dotted', linewidth=0.4)
fig_desire_expec_avg_group.set_xlabel('Student groups', size="medium", labelpad=14)
fig_desire_expec_avg_group.set_ylabel('Average grades', size = "medium", labelpad=20)
fig_desire_expec_avg_group.tick_params(axis='x', labelsize="small")
fig_desire_expec_avg_group.set_title(
    "Grade Averages per Group (Ward Clustering with PCA)", size=12, pad=12)

# Alter legend's placement
fig_desire_expec_avg_group.legend(
    loc="center left",
    fontsize = "x-small", # 'xx-small', 'x-small', 'small', 'medium'
    bbox_to_anchor = (1, 0.7)
)

fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/ward/"+"desire_exp_avg_row_ft_pca_ward"+".jpg",
    format="jpg", bbox_inches="tight", dpi = 200
)
fig_desire_expec_avg_group.get_figure().savefig(
    "./output/clustering/ward/"+"desire_exp_avg_row_ft_pca_ward"+".pdf",
    format="pdf", bbox_inches="tight"
);
# -

# We observe that PCA even worsens the ward clustering results. Indeed, we do not need PCA here; we can use all 6 row mean features!
