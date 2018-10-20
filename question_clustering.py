# Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from random import shuffle
from math import isnan
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from scipy.cluster.vq import kmeans,vq
from statistics import mode

from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import all_palettes
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score
from scipy.spatial.distance import pdist, euclidean


### Import dependencies and preprocess file ###

def preprocess_file(file_name, vector_column = 'q_text', label_column = '', delimiter = ',', encoding = 'latin-1'):
    '''
    Use: 
    Initial processing on the file column for Doc2Vec
    
    Arguments:
    file_name: file whose column needs to be processed
    vector_column: column whose values need to be processed as X
    label_column: column whose values need to be processed as y
    delimiter: symbol used to separate different fields in the file
    encoding: most csv files need latin-1 encoding for some reason
    
    Returns:
    file_df: the dataframe version of the file after preprocessing
    tagged_data: cleaned and tagged data for use in doc2vec
    data: fetched data
    labels: fetched and converted labels
    '''

    file_df = pd.read_csv(file_name, delimiter = delimiter, encoding = encoding)

    # NOTE: I would recommend that all sentences encoded be compiled in a single file

    data = []
    labels = []
    
    # deduplicate questions
    #file_df.drop_duplicates([vector_column], inplace = True)
    
    file_df['outcome'] = file_df['outcome'].apply(lambda x: '(' + str(x) + ')')
    
    # remove voided questions
    file_df = file_df[file_df.q_status != 'voided']
    
    # reset the index to prevent KeyError
    file_df = file_df.reset_index()

    for row in range(len(file_df)):
        # print(len(file_df))
        # print(file_df['options'][row])
        if file_df['options'][row][0] != "(":
            # print("here")
            index = file_df['options'][row].find(':')
            file_df[vector_column][row] = file_df[vector_column][row] + ' ' + file_df['options'][row][:index]
            file_df['options'][row] = file_df['options'][row][index + 2:].strip()
    
    # fetching data
    for sentence in file_df[vector_column]:
        if vector_column != 'q_text':
            sentence = sentence[7:]
        data.append(sentence)

    # print(file_df)        
            
    # fetching labels
    if label_column:
        for result in file_df[label_column]:
            if result == 'buy':
                labels.append(1)
            elif result == 'sell':
                labels.append(0)
            else:
                pass
    
    # processing step using tokenization   
    tagged_data = [TaggedDocument(words = word_tokenize(_d.lower()), tags = [str(i)]) for i, _d in enumerate(data)]
    
    return file_df, tagged_data, data, labels

 ### (OPTIONAL) Training the model (I'm using the saved model) ###
 
def train_doc2vec(train_data, num_epochs = 200, vec_size = 20, alpha = 0.01, output_file = 'd2v_test.model'):
    '''
    Use:
    Train on the cleaned data to get vector representation 
    and save the model
    
    Arguments:
    tagged_data: cleaned data ready for use
    num_epochs: number of iterations for training
    vec_size: vector size (hyperparameter) for representation
    alpha = learning rate
    output_file = filename for the saved model after doc2vec
    
    Returns:
    It does not return anything but saves the model representation
    '''
    max_epochs = num_epochs
    vec_size = vec_size
    alpha = alpha
    
    # min_count: number of words
    model = Doc2Vec(size = vec_size,
                    alpha = alpha, 
                    min_alpha = 0.00025,
                    min_count = 1,
                    dm = 1)

    model.build_vocab(train_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        shuffle(train_data)
        model.train(train_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(output_file)
    print("Model saved in file {}".format(output_file))


### Generate Respective Question Embeddings ###

def generate_vectors(model_file):
    '''
    Use:
    Generates vectors after loading the trained model
    
    Arguments:
    model_file: saved trained model file to load
    
    Returns:
    vectors: generated representations for training set
    '''
    model= Doc2Vec.load(model_file)

    vectors = []
    for i in range(len(model.docvecs)):
        vectors.append(model.docvecs[i])
        
    return vectors

def create_question_embedding(questions_df, model_file):
    '''
    Use:
    Maps questions to their respective generated embeddings
    
    Arguments:
    questions_df: dataframe containing question data
    model_file: saved/generated trained model file on question data
    
    Returns:
    question_embedding: dictionary mapping question id to embedding
    '''
    question_embedding = {}
    vectors = generate_vectors(model_file)
    for index in range(len(questions_df)):
        question_id = questions_df['ifp_id'][index]
        question_embedding[question_id] = vectors[index]

    return question_embedding

### Cluster Questions into groups ###

def cluster_questions(q_embedding, n_clusters):
    '''
    Use:
    cluster the question embeddings into groups
    
    Arguments:
    q_embedding: embeddings of all the questions
    n_clusters: number of cluster groups
    
    Returns:
    q_id: all the question_id (order preserved)
    X: all the embeddings (order preserved)
    y: all the labels (order preserved)
    '''
    q_id = np.asarray(list(q_embedding.keys()))
    X = np.asarray(list(q_embedding.values()))
    pca = PCA(n_components = 2)

    X_reduced = pca.fit_transform(X)
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(X_reduced)
    y = kmeans.labels_

    return q_id, X, y

### Plot the cluster group representation ###

def plot_graph(X, y, q_id, model):
    '''
    Use:
    Plots the cluster groups representation
    
    Arguments:
    X: question embedding
    y: corresponding cluster group label
    q_id: question_ids
    model: technique for dimensionality reduction (pca or tsne)
    
    Returns:
    Corresponding plot of the graph
    '''
    if model == 'tsne':
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, init='pca', perplexity=30)
        # X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
        tsne_result = tsne_model.fit_transform(X)
        tsne_embedding = pd.DataFrame(tsne_result, columns=['x', 'y'])
        tsne_embedding['hue'] = y
        tsne_embedding['question_id'] = q_id



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
                    <span style="font-size: 12px; font-weight: bold;">Question ID:</span>
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
        # X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
        pca_result = pca.fit_transform(X)

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


### Main program of execution ###

def main():
	# preprocess the question data file
	questions_df, tagged_data, data, labels = preprocess_file("data/ifps.csv", "q_text")

	# experiment here with different hyperparameters
	# train_doc2vec(tagged_data, num_epochs = 200, vec_size = 20, alpha = 0.01)

	# using pre-saved model to generate embeddings
	question_embedding = create_question_embedding(questions_df, 'models/d2v_test.model')

	# define no of clusters and group the questions
	silhouette_score_list = []
	calinski_harabasz_score_list = []
	davies_bouldin_index_list = []
	X_list = []
	y_list = []
	qid_list = []

	# calculating cluster correctness evaluation metrics for all selected groupings
	n_clusters = list(range(2, 11))
	for n_cluster in n_clusters:	
		qid, X, y = cluster_questions(question_embedding, n_cluster)
		s_score = calculate_silhouette_score(X, y)
		silhouette_score_list.append(s_score)
		ch_score = calculate_calinski_harabasz_score(X, y)
		calinski_harabasz_score_list.append(ch_score)
		db_index = davies_bouldin(X, y)
		davies_bouldin_index_list.append(db_index)
		X_list.append(X)
		y_list.append(y)
		qid_list.append(qid)

	# calculating best indices according to the respective algorithms
	chosen_sscore_cluster_no = silhouette_score_list.index(max(silhouette_score_list))	
	chosen_chscore_cluster_no = calinski_harabasz_score_list.index(max(calinski_harabasz_score_list))			
	chosen_dbindex_cluster_no = davies_bouldin_index_list.index(min(davies_bouldin_index_list))
	try:
		chosen_index = mode([chosen_dbindex_cluster_no, chosen_chscore_cluster_no, chosen_sscore_cluster_no])
		apt_cluster_no = chosen_index + 2
		print("Appropriate Cluster Number: " + str(apt_cluster_no))
		# plot tsne and pca plots once appropriate cluster no found	
		plot_graph(X_list[chosen_index], y_list[chosen_index], qid_list[chosen_index], 'tsne')
		plot_graph(X_list[chosen_index], y_list[chosen_index], qid_list[chosen_index], 'pca')
	except:
		print("Silhouette scores: ", silhouette_score_list)
		print("Calinski-Harabasz scores: ", calinski_harabasz_score_list)
		print("Davies-Bouldin scores: ", davies_bouldin_index_list)
		print()
		print("Please note the best possible no of clusters")
		print("Higher the silhouette scores, the better")
		print("Higher the calinski_harabaz_score, the better")
		print("Lower the davies_bouldin index, the better")
		print()
		print("Please hard-code the appropriate value if neccesary.")
		return
	
	

if __name__ == '__main__':
	main()	