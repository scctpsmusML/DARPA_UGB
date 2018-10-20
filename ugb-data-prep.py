# Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle
from math import isnan
import pickle


def preprocess_file(file_name, vector_column, label_column = '', delimiter = ',', encoding = 'latin-1'):
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

def create_true_answer_embedding(questions_df, model_file):
    '''
    Use:
    Creates True Answer Embedding on the basis of trained doc2vec model

    Arguments:
    questions_df: dataframe containing question data
    model_file: saved/generated trained model file on question data

    Returns:
    true_answer_embedding: dictionary mapping true answer to a question to an embedding
    '''
    model = Doc2Vec.load(model_file)
    true_answer_embedding = {}
    for index in range(len(questions_df)):
        options = questions_df['options'][index].split(', (')
        for option in options:
            val = option.strip().split(' ', 1)
            if val[0][0] != '(':
                val[0] = '(' + val[0]
            if val[0] == questions_df['outcome'][index]:
                model.random.seed(0)
                true_answer_embedding[questions_df['ifp_id'][index]] = model.infer_vector(word_tokenize(val[1].strip().lower()), steps = 200)
                
    return true_answer_embedding                  

def fetch_collated_mapping(file_df):
    '''
    Use:
    Fetch correspondence between merged feature columns and individual feature columns

    Arguments:
    file_df: user description file dataframe

    Returns:
    inv_key_map: collated columns as keys to individual old keys as values in a list
    '''
    original_keys = list(file_df.columns)
    
    key_map = {}
    for original_key in original_keys:
        if original_key == 'userid_yr4':
            key_map[original_key] = 'user_id'
            continue
        if '_yr' in original_key:
            end = original_key.find('_y')
            # print(end)
            key_map[original_key] = original_key[:end]
        else:
            key_map[original_key] = original_key


    inv_key_map = {}

    for k, v in key_map.items():
        inv_key_map[v] = inv_key_map.get(v, [])
        inv_key_map[v].append(k)

    return inv_key_map

def fetch_cleaned_dataframe(file_df, key_map):
    '''
    Use:
    Transform the dataframe by collating redumdant columns using the column mapping

    Arguments:
    file_df: user description file dataframe
    key_map: column correspondence mapping
	
    Returns:
    user_df: transformed dataframe with collated relevant columns
    '''
    user_dict = {}
    list_keys = list(key_map.keys())
    
    for key in list_keys:
        temp = []
        mock_df = file_df[key_map[key]]
        for i in range(len(mock_df)):
            for idx, col in enumerate(list(mock_df.columns)[::-1]):
                if pd.notnull(mock_df[col][i]):
                    temp.append(mock_df[col][i])
                    break
                else:
                    if idx < (len(list(mock_df.columns)) - 1):
                        continue
                    else:
                        temp.append(np.nan)
                user_dict[key] = temp
        
    user_df = pd.DataFrame(user_dict)
    
    # find the columns where percentage of missing values is > 50%
    to_delete = []
    for element in list(user_df.columns):
        if user_df[element].isnull().sum() > 7000:
            to_delete.append(element)

    # drop those columns        
    for element in to_delete:
        user_df.drop([element], axis = 1, inplace=True)    
    
    return user_df

def detect_corrupt_columns(filename_df):
    '''
    Use:
    Find ill-formatted columns in the dataframe

    Arguments:
    filename_df: cleaned dataframe of user description info

    Returns:
    corrupt_columns: list of ill-formatted columns
    '''
    corrupt_columns = [x for x in filename_df.columns if filename_df[x].dtypes == 'object']
    return corrupt_columns

def fix_column(filename_df, column_name, action = None):
    '''
    Use:
    Fixes column formats based on actions needed

    Arguments:
    filename_df: cleaned dataframe of user description info
    column_name: corrupted column
    action: corrective measure for the column (found out by manual observation of columns)

    Returns:
    filename_df: formatted dataframe
    '''
    
    if action == 'fix_decimal':
        filename_df[column_name] = filename_df[column_name].replace(',', '.', regex = True)
        filename_df[column_name] = filename_df[column_name].astype('float64')
    elif action == 'fix_comma_numbers':
        filename_df[column_name] = filename_df[column_name].replace(',', '', regex = True)
        filename_df[column_name] = filename_df[column_name].astype('float64')
    else:
        print("Only 'fix_decimal' and 'fix_comma_numbers' can be used as action options")
        return
    
    if column_name == 'numeracy_1':
        format_list = filename_df[column_name].tolist()

        filename_df[column_name] = [1 if (str(format_list[index]) == '0.25' or str(format_list[index]) == '5.0') 
                                    else 0 if (isnan(float(format_list[index])) == False)
                                    else format_list[index] for index in range(len(format_list))]
        
    return filename_df


def populate_main_dictionary(forecast_df_list, questions_df, q_embedding, ta_embedding, imputed_df, model_file):
    '''
    Use:
    Build a dictionary combining forecasting information and personal attributes for a user

    Arguments:
    forecast_df_list: list of forecast dataframes over all the years
    questions_df: dataframe containing questions data
    q_embedding: dictionary mapping a question to its respective embedding
    ta_embedding: dictionary mapping a question to its true answer embedding
    imputed_df: imputed dataframe of user description data
    model_file: saved/generated trained model file on question data 
    
    Returns:
    ultimate_dictionary: unified data structure containing all the user information
    excluded_list: deviant users not having attributes and/or participated in forecasting
    '''

    # load the saved model to create user answer embedding 
    model = Doc2Vec.load(model_file)

    # main dictionary to be populated
    ultimate_dictionary = {}

    # user ids with no attributes but have answered questions need to be removed
    excluded_list = []

    # iterations through the files begin here
    for forecast_df in forecast_df_list:
        for row in range(len(forecast_df)):
            if forecast_df['q_status'][row] != 'closed':
                continue

            # adding user id as primary key
            user_id = forecast_df['user_id'][row]

            # skip rows for which user id is nan
            if isnan(user_id):
                continue

            # add user id as primary key    
            if user_id not in ultimate_dictionary:
                ultimate_dictionary[user_id] = dict()

            # adding question id as secondary key    
            question_id = forecast_df['ifp_id'][row]
            question_id_embedding = q_embedding[question_id]

            # adding the below keys as tertiary keys
            timestamp = forecast_df['timestamp'][row]
            user_answer = forecast_df['answer_option'][row]

            # fetch the equivalent index in the questions dataframe
            index = questions_df.index[questions_df.ifp_id == question_id].values[0]
            num_options = questions_df['n_opts'][index]

            # parsing the options and embedding the user answer accordingly
            options = questions_df['options'][index].split(', (')
            for option in options:
                val = option.strip().split(' ', 1)
                if val[0][0] != '(':
                    val[0] = '(' + val[0]
                # print(val)    
                if val[0] == user_answer:
                    model.random.seed(0)
                    user_answer_embedding = model.infer_vector(word_tokenize(val[1].strip().lower()), steps = 200)      


            probability = forecast_df['value'][row]
            true_answer = ta_embedding[question_id]


            # populating the main dictionary begins
            if question_id not in ultimate_dictionary[user_id]:
                ultimate_dictionary[user_id][question_id] = dict()
                ultimate_dictionary[user_id][question_id]['q_embedding'] = question_id_embedding
                ultimate_dictionary[user_id][question_id]['timestamps'] = [timestamp]
                ultimate_dictionary[user_id][question_id]['num_options'] = num_options
                ultimate_dictionary[user_id][question_id]['ua_embedding'] = [user_answer_embedding]
                ultimate_dictionary[user_id][question_id]['probability'] = [probability]
                ultimate_dictionary[user_id][question_id]['ta_embedding'] = true_answer
            else:
                ultimate_dictionary[user_id][question_id]['timestamps'].append(timestamp)
                ultimate_dictionary[user_id][question_id]['ua_embedding'].append(user_answer_embedding)
                ultimate_dictionary[user_id][question_id]['probability'].append(probability)

            # finding the equivalent index in the imputed user description dataframe
            index = imputed_df.index[imputed_df.user_id == user_id]

            # adding other secondary keys based on imputed user description dataframe
            for col in list(imputed_df.columns):
                if index.any():
                    ultimate_dictionary[user_id][col] = imputed_df[col][index[0]]
                else:
                    excluded_list.append(user_id)
                    break
    
    ultimate_dictionary = {k: ultimate_dictionary[k] for k in ultimate_dictionary if not isnan(k)}
    
    return ultimate_dictionary, excluded_list

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


def main():
	''' Main Code of Execution '''
	
	# preprocess the questions file to fetch questions dataframe
	questions_df, tagged_data, data, labels = preprocess_file("data/ifps.csv", "q_text")

	# optional: train your own model
	# train_doc2vec(tagged_data, num_epochs = 200, vec_size = 20, alpha = 0.01)

	# using pre-saved trained doc2vec model to generate embeddings
	question_embedding = create_question_embedding(questions_df, 'models/d2v_test.model')
	model_file = "models/d2v_test.model"

	# create true answer embedding
	true_answer_embedding = create_true_answer_embedding(questions_df, 'models/d2v_test.model')

	# open the user description file for processing
	all_differences_df = pd.read_csv('data/all_individual_differences.tab', delimiter='\t')

	# find mapping of redundant columns
	inv_key_map = fetch_collated_mapping(all_differences_df)

	# merge the redundant columns and remove columns with > 50% values missing
	d_df = fetch_cleaned_dataframe(all_differences_df, inv_key_map)

	# find deviant columns
	corrupt_columns = detect_corrupt_columns(d_df)

	# corrupt columns are ['newshrs', 'crt_2', 'crt_3','numeracy_1']
	d_df = fix_column(d_df, 'newshrs', 'fix_decimal')
	d_df = fix_column(d_df, 'newshrs', 'fix_comma_numbers')
	d_df = fix_column(d_df, 'numeracy_1', 'fix_comma_numbers')

	# dropping all crt based columns as we have redundant columns in crt_1_cor, crt_2_cor, crt_3_cor
	# d_df.drop(['crt_1', 'crt_2', 'crt_3'], axis = 1, inplace = True)

	# optional step to save this file to use for further imputation and lda
	# d_df.to_csv('./user_description_formatted.csv')

	# use the imputed file to eventually populate the dictionary with user attributes
	ud_df = pd.read_csv('data/user_description_formatted____amelia_noms_it_100_196.csv')
	ud_df.drop(['Unnamed: 0'], axis = 1, inplace=True)
	ud_df['user_id'] = d_df['user_id']

	# setting up the forecast survey files for parsing into dictionary
	forecast_y1_df = pd.read_csv("data/survey_fcasts.yr1.tab", delimiter = '\t')
	forecast_y1_df['answer_option'] = forecast_y1_df['answer_option'].apply(lambda x: '(' + str(x) +')')

	forecast_y2_df = pd.read_csv("data/survey_fcasts.yr2.tab", delimiter = '\t')
	forecast_y2_df['answer_option'] = forecast_y2_df['answer_option'].apply(lambda x: '(' + str(x) +')')

	forecast_y3_df = pd.read_csv("data/survey_fcasts.yr3.tab", delimiter = '\t')
	forecast_y3_df['answer_option'] = forecast_y3_df['answer_option'].apply(lambda x: '(' + str(x) +')')

	forecast_y4_df = pd.read_csv("data/survey_fcasts.yr4.tab", delimiter = '\t')
	forecast_y4_df['answer_option'] = forecast_y4_df['answer_option'].apply(lambda x: '(' + str(x) +')')

	forecast_dataframe_list = [forecast_y1_df, forecast_y2_df, forecast_y3_df, forecast_y4_df]

	ultimate_dictionary, excluded_list = populate_main_dictionary(forecast_dataframe_list, questions_df, question_embedding, true_answer_embedding, ud_df, model_file)

	# save the generated dictionary (python object)
	save_object(ultimate_dictionary, 'data/all_yrs_data_complete_cleaned')



if __name__ == '__main__':
    main()