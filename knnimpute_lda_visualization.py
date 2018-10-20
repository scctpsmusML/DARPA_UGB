import pandas as pd
import numpy as np
import pickle
#from ggplot import *
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes

from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)

import lda

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Normalizer


def knn_reimpute(input_file, imputed_files):
    '''
    Use:
    Check user attributes data, imputations and reimpute poorly imputed data

    Arguments:
    input_file: pre-imputed file obtained while preparing data (path)
    imputed_files: list of imputed files used to reimpute using KNN-Impute (paths)

    Returns:
    knn_df: reimputed user info dataframe
    '''
    tmp = []
    for imputed_file in imputed_files:
        df = pd.read_csv(imputed_file)
        tmp.append(df.values)

    cols = list(df.columns.values)
    tmp = np.array(tmp)

    df = pd.read_csv(input_file)

    min_val = df.min()
    max_val = df.max()

    df.drop(['user_id', 'crt_1', 'crt_2', 'crt_3'], axis = 1, inplace = True)

    x = np.nanmedian(tmp, 0)
    # print(x.shape)
    df = pd.DataFrame(data = x, columns = cols)

    for col in cols:
    	df[col] = df[col].mask(df[col].le(0))

    # replace missing data with knn
    # print('imputing with K-Nearest Neighbors')
    x = df.values
    missing_mask = pd.isnull(x)
    data_knn = knn_impute_reference(x.copy(), missing_mask, k=5)

    udf_amelia_df = pd.read_csv(imputed_files[-1])
    knn_df = pd.DataFrame(data_knn, columns = list(udf_amelia_df.columns))
    
    if 'Unnamed: 0' in set(knn_df.columns): 
        knn_df = knn_df.drop(['Unnamed: 0'], axis = 1)
    
    udf_df = pd.read_csv(input_file)
    knn_df['user_id'] = udf_df['user_id']
    
    return knn_df

def save_object(obj, name):
    '''
    Use:
    Save any Python object as a pickle file for future use

    Arguments:
    obj: Python object
    name: Name of the saved file

    Returns:
    Nothing, but a file should be created with name 'name' containing 'obj'
    '''
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def run_lda(file_df, name):
    '''
    Use:
    Performs LDA on the data to group users into clusters based on latent factors

    Arguments:
    file_df: input user info dataframe
    name: Name of the saved results file

    Returns:
    Nothing, but a file should be created with name 'name' containing lda_results
    '''
    cleaned_data = file_df.values.astype('int64')

    lda_results = []

    topics = [2, 3, 4, 5, 6]

    for n_topics in topics:
        lda_model = lda.LDA(n_topics=n_topics, n_iter = 300)
        X = lda_model.fit_transform(cleaned_data)
        lda_results.append(X)
    
    save_object(lda_results, name)     

def feature_selection_for_cluster_results(lda_result, user_info_df, cluster_no):
    '''
    Use:
    Displays important features for a cluster in a group of clusters of users

    Arguments:
    lda_result: lda_result computed for a particular no of clusters
    user_info_df: user info dataframe
    cluster_no: label of a particular cluster of interest

    Returns:
    feature_names: names of features important for that user cluster among clusters
    '''
    cluster_labels = lda_result.argmax(axis = 1)
    user_info_df['labels'] = cluster_labels

    user_info_df_filtered = user_info_df.loc[user_info_df['labels'] == cluster_no]

    labels = user_info_filtered['labels'].tolist()
    user_info_df_filtered = user_info_df_filtered.drop(['labels'], axis = 1)

    X, y = user_info_df_filtered, labels

    feature_selector = SelectKBest(chi2, k=4)
    feature_selector.fit(X, y)

    idxs_selected = feature_selector.get_support(indices=True)
    feature_names = []
    for idx in idxs_selected:
        feature_names.append(user_info_df_filtered.columns.tolist()[idx])

    # print(feature_names) 
    
    features_dataframe_new = user_info_df_filtered[feature_names]
    # below code used for analyses
    # features_dataframe_new.describe()
    # features_dataframe_new.hist()
    # feature_max = features_dataframe_new.max(axis = 1)
    # normalized_df = features_dataframe_new.divide(feature_max, axis = 0)
    # normalized_df.hist()

    return feature_names

def plot_graph(X_df, y, u_id, model):
    '''
    Use:
    Plots the cluster groups representation
    
    Arguments:
    X_df: user info dataframe
    y: corresponding cluster group label
    u_id: user_ids
    model: technique for dimensionality reduction (pca or tsne)
    
    Returns:
    Corresponding plot of the graph
    '''

    rng = np.random.RandomState(42)
    outliers_fraction =  0.01

    clf = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
    y_pred = clf.fit_predict(X_df.values)
    scores_pred = clf.negative_outlier_factor_

    outliers_rem = np.where(y_pred == 1)

    if model == 'tsne':
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, init='pca', perplexity=30)
        tsne_result = tsne_model.fit_transform(Normalizer().fit_transform(X_df.values[outliers_rem]))
        tsne_embedding = pd.DataFrame(tsne_result, columns=['x', 'y'])
        tsne_embedding['hue'] = y
        tsne_embedding['user_id'] = u_id



        source = ColumnDataSource(
                data=dict(
                    x = tsne_embedding.x,
                    y = tsne_embedding.y,
                    colors = [all_palettes['Set1'][8][i] for i in tsne_embedding.hue],
                    question = tsne_embedding.question_id,
                    cluster_no = tsne_embedding['hue'],        
                    alpha = [0.9] * tsne_embedding.shape[0],
                    size = [7] * tsne_embedding.shape[0]
                )
            )

        hover_tsne = HoverTool(names=["df"], tooltips="""
            <div style="margin: 10">
                <div style="margin: 0 auto; width:300px;">
                    <span style="font-size: 12px; font-weight: bold;">User ID:</span>
                    <span style="font-size: 12px">@question</span>
                    <span style="font-size: 12px; font-weight: bold;">Cluster no:</span>
                    <span style="font-size: 12px">@cluster_no</span>
                </div>
            </div>
            """)

        tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
        plot_tsne = figure(plot_width=700, plot_height=700, tools=tools_tsne, title='tSNE (2 components) on question clusters')
        plot_tsne.circle('x', 'y', size='size', fill_color='colors', 
                         alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df")

        layout = column(plot_tsne)
        show(layout)
    
    elif model == 'pca':
        plt.clf()
        pca = PCA(n_components = 2)
        pca_result = pca.fit_transform(Normalizer().fit_transform(X_df.values[outliers_rem]))

        cluster_no = y    # Labels of cluster 0 to 3

        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_no, s=50)

        ax.set_title('PCA (2 component) on {} cluster groups'.format(len(set(cluster_no))))
        plt.colorbar(scatter)

        print('Cumulative explained variation for 2 principal components: {}'.format((pca.explained_variance_ratio_)))

        fig.show()
        plt.savefig('pca_plot_for_{}_clusters'.format(len(set(cluster_no))))
        
    else:
        print("The model could only be 'tsne' or 'pca'. Please check your model of choice.")
        return

### Evaluation Metrics ###

def davies_bouldin(X, labels):
    '''
    Use:
    Calculates the Davies-Bouldin index for clustered data

    Arguments:
    X: Input data
    labels: cluster labels for the input data

    Returns:
    dbi: Davies-Bouldin Index for the data
    '''
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [(np.mean(k, axis = 0)).astype(float) for k in cluster_k]

    # calculate cluster dispersion
    S = [np.mean([euclidean((p), centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    Ri = []

    for i in range(n_cluster):
        Rij = []
        # establish similarity between each cluster and all other clusters
        for j in range(n_cluster):
            if j != i:
                r = np.float((S[i] + S[j]) / (euclidean(centroids[i], centroids[j]) + 10**(-8)))
                Rij.append(r)
         # select Ri value of most similar cluster
        Ri.append(max(Rij))

    # get mean of all Ri values    
    dbi = np.mean(Ri)
    
    dbi = np.asarray(Ri, dtype=float).mean()

    return dbi

def calculate_silhouette_score(X, labels):
    '''
    Use:
    Calculates the average silhouette score for clustered data

    Arguments:
    X: Input data
    labels: cluster labels for the input data

    Returns:
    silhouette_avg: Average silhouette score for the data
    '''
    silhouette_avg = silhouette_score(X, labels)
    return silhouette_avg

def calculate_calinski_harabasz_score(X, labels):
    '''
    Use:
    Calculates the average calinski-harabasz score for clustered data

    Arguments:
    X: Input data
    labels: cluster labels for the input data

    Returns:
    ch_avg: Average calinksi harabasz score for the data
    '''
    ch_avg = calinski_harabaz_score(X, labels)
    return ch_avg

def evaluate_cluster_groups(lda_results, knn_df_cleaned):
    '''
    Use:
    evaluate cluster groupings formed by lda to determine the best cluster grouping

    Arguments:
    lda_results: list of all the lda results considering all possible groupings
    knn_df_cleaned: dataframe of users having answered questions and attributes 
    
    Returns:
    silhouette_scores: Silhouette scores for all possible cluster groups
    ch_scores: Calinski-Harabasz indices for all possible cluster groups
    db_scores: Davies-Bouldin scores for all possible cluster groups
    '''
    rng = np.random.RandomState(42)
    outliers_fraction =  0.01

    clf = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
    y_pred = clf.fit_predict(X_df.values)
    scores_pred = clf.negative_outlier_factor_

    outliers_rem = np.where(y_pred == 1)
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    for i in range(len(lda_results)):
        X = Normalizer().fit_transform(knn_df_cleaned.values[outliers_rem])
        cluster_labels = np.array(lda_results[i][outliers_rem].argmax(axis = 1))
        sil_avg = calculate_silhouette_score(X, cluster_labels)
        ch_avg = calculate_calinski_harabasz_score(X, cluster_labels)
        db_avg = davies_bouldin(X, cluster_labels)
        print("For " + str(i + 2) + " clusters, avg. silhouette score = " + str(sil_avg))
        print("For " + str(i + 2) + " clusters, avg. calinski harabaz score = " + str(ch_avg))
        print("For " + str(i + 2) + " clusters, avg. davies bouldin score = " + str(db_avg))
        silhouette_scores.append(silhouette_avg)
        ch_scores.append(ch_avg)
        db_scores.append(db_avg)

    return silhouette_scores, ch_scores, db_scores    

def create_labeled_file(cleaned_df, knn_df_cleaned, lda_result, name):
    '''
    Use:
    Created Labeled User Description files based on cluster assignment and no of clusters

    Arguments:
    cleaned_df: All user cleaned info dataframe
    knn_df_cleaned: users having answered questions and attributes dataframe
    lda_result: labels for user association 
    name: name of the labeled file created

    Returns:
    Nothing. But a labeled file should be created on running it.
    '''
    knn_df_cleaned['labels'] = lda_result.argmax(axis = 1)
    clusters = []
    for row in range(len(cleaned_df)):
        if cleaned_df['user_id'][row] in set(knn_df_cleaned['user_id']):
            clusters.append(knn_df_cleaned['labels'][knn_df_cleaned.index[knn_df_cleaned.user_id == cleaned_df['user_id'][row]][0]])
        else:
            clusters.append(np.nan)

    cleaned_df['labels'] = clusters        
    cleaned_df.drop(['crt_1', 'crt_2', 'crt_3'], axis = 1, inplace=True)
    cleaned_df.to_csv(name)

def main():
    imputed_files = ['data/user_description_formatted____amelia_noms_it_100_196.csv',
                    'data/user_description_formatted____amelia_noms_it_100_197.csv',
                    'data/user_description_formatted____amelia_noms_it_100_198.csv',
                    'data/user_description_formatted____amelia_noms_it_100_199.csv',
                    'data/user_description_formatted____amelia_noms_it_100_200.csv']

    input_file = 'data/user_description_formatted.csv'

    knn_df = knn_reimpute(input_file, imputed_files)
	
	#with open('data/test_all_yrs_data_complete_cleaned.pkl', 'rb') as f:
    with open('data/all_yrs_data_complete_cleaned.pkl', 'rb') as f:
        cleaned_dictionary = pickle.load(f)

    knn_df_cleaned = knn_df.loc[knn_df['user_id'].isin(set(cleaned_dictionary.keys()))]
    knn_df_cleaned = knn_df_cleaned.reset_index().drop(['index'], axis = 1)
    
    run_lda(knn_df_cleaned, 'data/lda_results_formatted')

    # using pre-saved lda results instead
    with open('data/lda_results_formatted.pkl', 'rb') as f:
        lda_results = pickle.load(f)
    
    sil_scores, ch_scores, db_scores = evaluate_cluster_groups(lda_results, knn_df_cleaned)
    create_labeled_file(knn_df, knn_cleaned_df, lda_results[0], 'test.csv')

if __name__ == '__main__':
    main()