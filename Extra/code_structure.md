## The Structure of the Program Code

- The folder `"data"` is used to load / store data sets. Output images, reports and other documents are stored in the git-ignored folder `"output"`. Several images are gathered in the folder `"readme_images"` for isolated usage in this readme file.

- The **1st** piece of code is `"010_data_preparation.py"`; here we import the raw data from `"raw_data.csv"` and `"raw_data_new_SS2023.csv"`, fix the encoding and replace the long column descriptions by abbreviations / question codes, export the question abbreviations, remove 6 rows with all NAs and export the data as `"data_merged.xlsx"` / `"data_merged.csv"` for further processing, as well as we create the first report on the data using Pandas Profiling (ydata-profiling).

- The **2nd** piece of code, `"Extra/020_presentation_1.ipynb"`, is a small jupyter-based HTML presentation (powered by [Reveal.js](https://revealjs.com/)), which gives us the first impression on the data (surveyed specializations, duplicates, missing data, outliers, correlations, regression plot).

- The **3rd** piece of code, `"030_reduce_impute_data.py"` prepares and exports the reduced data set `"data_reduced.xlsx"` by dropping all the rows with missing observations, as well as the imputed data sets `"data_imputed.xlsx"`  and `"data_imputed.csv"`.

- In the **4th** piece of code, `"040_specializations_and_averages.py"`, we compute, export (`"data_averages.csv"`) and plot the averages for all 24 SELAQ features. Then we compare the averages per specialization. The row averages for the question subgroups *Data Protection, LA General Functionality, LA Teacher-Related Features* are computed, visualized and exported as `"row_mean_features.csv"`. A corresponding HTML report is also generated (`"acceptance_report_row_mean_features.html"`). In the **5th** piece of code, `"045_specializations_and_averages.py"', we plot a bar chart for averages with error bars.

- The **6th** piece of code, `"050_distributions_and_correlations.py"`, visualizes the imputed data. We plot and export all the histograms, a heatmap and scatterplots for most correlated variables.

- In the **7th** piece of code, `"060_k_means_without_pca.py"`, we perform **K-Means clustering** without PCA. We apply the *Elbow* and *Silhouette* methods in YellowBricks library to evaluate the performance of K-Means and to choose the best cluster number $K=4$. The poor performance indicates the necessity of PCA. Therefore running this file is not necessary for our main results, we keep it just for the sake of completeness.

- In the **8th** piece of code, `"070_PCA_and_Kmeans.py"`, we perform **The principal component analysis (PCA)** followed by **K-Means clustering**. The PCA enables us to reduce the data dimension to 3. By examining the PCA coefficients, we observe that most students find the data security important, but they discourage the usage of their data. We apply the *Elbow* and *Silhouette* methods in YellowBricks library to evaluate the performance of K-Means and to choose the best cluster number $K=4$. The performance improves with PCA. We check the clusters by means of 3d plotly plots. Then we plot feature averages per group (first as a line plot for all questions, then as a bar plot for 3 question groups and per cluster). We also plot all the histograms per cluster. We reveal 4 important student groups; we call them *Enthusiast, Realist, Cautious, Indifferent*. This piece has important results and, as we will see later, other clustering algorithms perform worse!

- The **9th** piece of code, `"080_Kmeans_with_avg_features.py"`, is the inspirational continuation of the K-Means cluster analysis from `"090_PCA_and_Kmeans.py"`. Earlier we have generated 4 groups enabling explanations by means of our row mean features. Now we take these features as a starting point and perform K-Means clustering. The obtained results are comparable to the preceding ones.

- The **10th** piece of code, `"090_manual_clustering.py"`, is an interactive manual clustering environment enabling us to generate clusters with above-mentioned patterns manually and to view their structure. The results comply with the preceding ones.

- The **11th** piece of code, `"100_Hypotheses_plots.py"`, poses and confirms some interesting hypotheses (revealed by means of the K-Means cluster analysis). The animated plots show that students assessing one of 24 SELAQ questions highly give typically high grades for all 24 questions; we observe the largest effect size for questions 2.8, 2.14, 2.20. The same thing holds for low assessments; it is clearly expressed for question 2.3 and partially for 2.19 (in the last case, we observe considerable influence only on 2.21 and 2.22).

- In the next pieces of code, `"110xxx-130xxx"`, we try other clustering algorithms for our data. The only working algorithm here is the hierarchical (ward) clustering, but the results are not remarkable. We leave these code pieces for the sake of completeness.

- Some frequently used auxiliary functions and variables are included in `"functions_and_variables.py"`.

---
