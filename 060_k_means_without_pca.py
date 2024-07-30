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

# + [markdown] slideshow={"slide_type": "slide"}
# # Acceptance Study: K-Means cluster analysis without PCA
#
# In this piece of code, we perform **K-Means clustering** without PCA. We apply the *Elbow* and *Silhouette* methods in YellowBricks library to evaluate the performance of K-Means and to choose the best cluster number $k=4$. The poor performance indicates the necessity of PCA. Therefore running this file is not necessary for our main results, we keep it just for the sake of completeness.
# -

# ## Import packages and data

# ⚙️ Install the libraries if necessary

# +
# # !pip install yellowbrick
# # !pip install --upgrade yellowbrick
# -

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from functions_and_variables import plot_df_histograms
from math import ceil
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score, silhouette_samples

# +
# Use these 2 lines if you change the loaded functions during your work
# # %load_ext autoreload
# # %autoreload 2

# Appendix (run on demand)
# import matplotlib.cm as cm # color maps, resolves font problems etc
# from sklearn.metrics import silhouette_samples
# import os

# +
# Load imputed data
acceptance_imputed_df = pd.read_excel('./data/clean_data/data_imputed.xlsx',index_col="SurveyID")

# # A two-line way to load the data:
# acceptance_imputed_df = pd.read_excel('./data/clean_data/data_imputed.xlsx')
# # Set "SurveyID" as index column
# acceptance_imputed_df.set_index("SurveyID", inplace = True)

# + [markdown] slideshow={"slide_type": "slide"}
# Note: See "functions_and_variables.py" for the names of the surveyed specializations
# -

# ## K-Means clustering

# ### Feature preparation

# Select clustering features
acceptance_cluster_df = acceptance_imputed_df.copy()
acceptance_cluster_df = acceptance_cluster_df.iloc[:,4:] # drop the first 4 columns
# Optional: drop the last column (outlier question)
acceptance_cluster_df = acceptance_cluster_df.iloc[:,:-1]
acceptance_cluster_df.head(3)

# ### Evaluations with YellowBricks

# 📚 We use a nice ***Data Visualisation*** library dedicated to Machine Learning algorithms which is called [**`YellowBricks`**].

# #### The Elbow Method

# Try to find the Elbow of the KMeans algorithm on `acceptance_cluster_df` using the ***KElbowVisualizer***. Note that KMeans is stochastic (the results may vary even if we run the function with the same inputs' values). Hence, we specify a value for the `random_state` parameter in order to make the results reproducible.

# +
# Instantiate the clustering model and visualizer
kmeans_model = KMeans(random_state=42)
elbow_visualizer = KElbowVisualizer(kmeans_model, k=(2,9))
# elbow_visualizer = KElbowVisualizer(kmeans, n_clusters = (2,8))

elbow_visualizer.fit(acceptance_cluster_df) # Fit the data to the visualizer
elbow_plot = elbow_visualizer.poof() # or .show() # Finalize and render the figure
elbow_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_without_pca/"+"elbow_method"+".pdf",
    format="pdf", bbox_inches="tight");
elbow_plot.get_figure().savefig(
    "./output/clustering/kmeans/direct_without_pca/"+"elbow_method"+".jpg",
    format="jpg", bbox_inches="tight");
# -

# 👉 This `KElbowVisualizer` was able to detect the "elbow" at $K = 4$.

# #### The Silhouette Method

# +
range_n_clusters = list(range(2,9)) # [2, 3, 4, 5, 6, 7, 8]
num_clusters = len(range_n_clusters)
X = acceptance_cluster_df.to_numpy() # to get rid of warnings

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

plt.savefig("./output/clustering/kmeans/direct_without_pca/"+"silhouette_scores"+".pdf",
            format="pdf", bbox_inches="tight");
plt.savefig("./output/clustering/kmeans/direct_without_pca/"+"silhouette_scores"+".jpg",
            format="jpg", bbox_inches="tight");
# https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam
# -

# ####  The Silhouette Method: Visualization using silhouette samples

# +
X = acceptance_imputed_df
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    # The silhouette plot: The silhouette coefficient can range from -1, 1
    #  but in this example all lie within [-0.2, 0.6]
    ax1.set_xlim([-0.2, 0.5])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 42 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    clusterer.fit(X)
    cluster_labels_tmp = clusterer.labels_
    
    # Sorting clusters by size
    acceptance_imputed_df_tmp = acceptance_imputed_df.copy()
    acceptance_imputed_df_tmp["cluster_no"] = cluster_labels_tmp
    label_conv_dict = list(acceptance_imputed_df_tmp["cluster_no"].value_counts().index)
    
    # Dictionary to set new cluster labels
    label_conv_dict = dict(zip(label_conv_dict,sorted(label_conv_dict)))
    # Label convertion
    cluster_labels = acceptance_imputed_df_tmp['cluster_no'].map(label_conv_dict)    
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, -0.1, 0, 0.2, 0.4, 0.5])

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=15,
        fontweight="bold",
    )

    plt.savefig(
        "./output/clustering/kmeans/direct_without_pca/"+"silhouette_scores_"+str(n_clusters)+"_clusters.jpg",
        format="jpg", bbox_inches="tight"
    );
    
plt.show()
# -

# ### Histograms for 4 K-Means clusters

# +
# Clustering
n_clusters = 4
kmeans_scaler = KMeans(n_clusters = n_clusters, random_state=42) # reproducibility
kmeans_scaler.fit(acceptance_cluster_df)

# Getting cluster numbers
kmeans_labels = kmeans_scaler.labels_
acceptance_imputed_df["cluster_no"] = kmeans_labels

# Sorting clusters by size
kmeans_labels_by_size = list(acceptance_imputed_df["cluster_no"].value_counts().index)
# Dictionary to set new cluster labels
label_conv_dict = dict(zip(kmeans_labels_by_size,sorted(kmeans_labels_by_size)))
# Label convertion
acceptance_imputed_df['cluster_no'] = acceptance_imputed_df['cluster_no'].map(label_conv_dict)
# Check clusters
# acceptance_imputed_df["cluster_no"].value_counts()

# +
# %%time
# takes 23 sec
acceptance_df_cluster_list = [acceptance_imputed_df[
    acceptance_imputed_df["cluster_no"]==i].iloc[:,:-1] for i in range(n_clusters)]
# acceptance_df_cluster_list = [acceptance_imputed_df[
#     acceptance_imputed_df["cluster_no"]==i].iloc[:,:-1].iloc[:,4:] for i in range(5)]

for idx, df in enumerate(acceptance_df_cluster_list):
    suptitle = f"K-Means: Group {idx+1} of {n_clusters}, {len(df)} students of {len(acceptance_imputed_df)}"
    plot_df_histograms(
        df,
        output_name="clustering/kmeans/direct_without_pca/histograms_cluster_"+str(idx+1),
        suptitle = suptitle
    )
