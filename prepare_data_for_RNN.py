import os
import torch
import numpy as np
import sys
from itertools import repeat
import pickle
from datetime import datetime
import time
import collections


ANS_EMBEDDING_DIM = 20
null_embedding = np.full(ANS_EMBEDDING_DIM, -1)

def read_input():
    with open("data/all_yrs_data_complete_cleaned.pkl", "rb") as input_file:
        data = pickle.load(input_file)
    '''all_keys = data.keys()
    for key in all_keys:
        if np.isnan(key):
            print(key)'''
    return data
    
def investigate_data_structure(data):
    keys_1 = list(data.keys())
    vals_1 = data[keys_1[0]]
    
    keys_2 = list(vals_1.keys())
    vals_2 = vals_1[keys_2[0]]
    
    keys_3 = list(vals_2.keys())
    vals_3 = vals_2[keys_3[0]]
    
    #print((keys_1)) #user_id
    #print((keys_2)) #question_id
    #print((keys_3)) # ['q_embedding', 'timestamps', 'num_options', 'ua_embedding', 'probability', 'ta_embedding']
    
    #for keys_opts in keys_3:        
    #    print(keys_opts, ' ------> ', vals_2[keys_opts], '  type  ', type(vals_2[keys_opts]))


def get_all_questions(data):
    max_options =  -sys.maxsize -1
    keys_1 = list(data.keys())
    #print(keys_1)
    for key_1 in keys_1:
        vals_1 = data[key_1]
        keys_2 = list(vals_1.keys())
        #print(keys_2)
        for key_2 in keys_2:
            #print('key_2 ----->   ', key_2)
            vals_2 = vals_1[key_2]
            if isinstance(vals_2,dict):
                keys_3 = list(vals_2.keys())
            else:
                continue
            #print(keys_3)
            for key_3 in keys_3:
                vals_3 = vals_2[key_3]
                if key_3 == 'num_options':
                #print(key_3, ' ------> ', vals_2[key_3], '  type  ', type(vals_2[key_3]))
                #print()
                    #print(key_3, ' has ', vals_2[key_3], ' options')
                    max_options = max(max_options, vals_2[key_3])
    return max_options

def find_num_duplicate_object(list_temp_dict, key_1):
    #print((key_1))
    
    if key_1 == 600.0:
        for i in list_temp_dict:
            print(i.keys())
    '''for i in range(len(list_temp_dict)-1):
        for j in range(i+1, len(list_temp_dict)): 
            if list_temp_dict[i] == list_temp_dict[j]:
                print('equal')'''
                

def sort_dict_by_ts(temp_dict):
    temp_dict_keys = list(temp_dict.keys())
    temp_dict_values = list(temp_dict.values())
    temp_dict_keys.sort(key=lambda tup: tup[1])
    dict_ordered = collections.OrderedDict()
    for key in temp_dict_keys:
        dict_ordered[key] = temp_dict[key]
    return dict_ordered

def prepare_data_for_RNN(data, max_options):
    dict_RNN = {}
    nonparticipant_set = set([])
    participant_set = set([])
    all_set = set([])
    keys_1 = list(data.keys()) 
    loop_counter = 0
    for key_1 in keys_1:
        loop_counter+=1
        #if loop_counter%100==0:
            #print('*', end=' ')
        #print('user id : ', key_1)
        if np.isnan(key_1):
            #print('user id : ', key_1)
            continue
        all_set.add(key_1)
        #print('user_id : ', key_1)
        vals_1 = data[key_1]
        keys_2 = list(vals_1.keys())
        list_temp_dict = []
        for key_2 in keys_2: 
            #print('question_id : ', key_2)    
            
            vals_2 = vals_1[key_2]
            if isinstance(vals_2,dict):
                participant_set.add(key_1)
                keys_3 = list(vals_2.keys())
            else:
                #print('non-participant ', key_1)
                nonparticipant_set.add(key_1)
                continue
            number_of_options = vals_2['num_options']
            timestamps = vals_2['timestamps']
            
            #convert timestamp string to timestamp objects
            timestamps_datetime_objs = []
            for datets_str in timestamps:
                timestamps_datetime_objs.append(datetime.strptime(datets_str, "%Y-%m-%d %H:%M:%S"))
            
            #print('timestamp is stored as ', (timestamps_datetime_objs))
            question_embeddings = vals_2['q_embedding']
            user_ans_embeddings = vals_2['ua_embedding']
            true_ans_embeddings = vals_2['ta_embedding']
            probability = vals_2['probability']
            #print('number_of_options : ', number_of_options, 'timestamps : ', timestamps, 'question_embeddings : ', question_embeddings, 'user_ans_embeddings : ', user_ans_embeddings, 'probability : ', probability)
            #print()
            
            temp_dict = {}
            for ts in range(0, len(timestamps_datetime_objs), number_of_options):                
                new_timestamp = timestamps_datetime_objs[ts]
                question_id = key_2
                new_ts_q_id_pair = (question_id, new_timestamp)  
                #print('new_timestamp ', new_timestamp)              
                new_probability = probability[ts:ts+number_of_options] #start at index ts, go upto index ts+number_of_options but dont include ts+number_of_options meaning for ts=0 and number_of_options=4 get 0,1,2,3 index positions
                #if max_options>number_of_options:
                new_probability.extend(repeat(0, max_options - number_of_options)) #don't need to check because extend takes care of it
                new_user_answer_embeddings = user_ans_embeddings[ts:ts+number_of_options]
                new_user_answer_embeddings.extend(repeat(null_embedding, max_options - number_of_options))
                prob_uaembedding_qembedding = (new_probability, new_user_answer_embeddings, question_embeddings, true_ans_embeddings) 
                #print('timestamps : ', new_timestamp, 'new_ts_q_id_pair : ', new_ts_q_id_pair, 'new_probability : ', new_probability, 'prob_uaembedding_qembedding : ', prob_uaembedding_qembedding)                 
                temp_dict[new_ts_q_id_pair] = prob_uaembedding_qembedding 
                temp_dict = sort_dict_by_ts(temp_dict)
            if temp_dict not in list_temp_dict:    
                list_temp_dict.append(temp_dict)   
            #print('list_temp_dict size : ', len(list_temp_dict))        
                #print(temp_dict)
            #sort dictionary
        #print(len(list_temp_dict))
        #find_num_duplicate_object(list_temp_dict, key_1)
        
        dict_RNN[key_1] =  list_temp_dict   
        #print('dict_RNN : ', dict_RNN[key_1])    
    return dict_RNN, nonparticipant_set, participant_set, all_set            

#call all functions 
data = read_input()


max_options = get_all_questions(data)

#declare true answer probability matrix
#prob_matrix = np.zeros(-1, max_options)
dict_RNN, nonparticipant_set, participant_set, all_set = prepare_data_for_RNN(data, max_options)

#print('max_options : ', max_options)

with open('data/dict_RNN.pickle', 'wb') as handle:
    pickle.dump(dict_RNN, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''print(len(list(nonparticipant_set)))   
print(len(list(participant_set)))  
print(len(list(all_set))) '''  
print('Data preparation for RNN completed !!!')