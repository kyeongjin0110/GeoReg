import os
import re
import json
import time
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm

import configparser
from argparse import Namespace
from sklearn.preprocessing import StandardScaler

import openai
# from arcgis.gis import GIS
from easyllm.clients import huggingface

def read_config(
    config_path='./config_ours.ini'
):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if len(config.sections()) == 0:
        return None
    
    ns_config = Namespace()
    for section in config.sections():
        setattr(ns_config, section, Namespace())
        for attr, val in config[section].items():
            setattr(getattr(ns_config, section), attr, val)
    
    return ns_config

# def login_gis_portal(
#     api_key,
#     url="https://www.arcgis.com"
# ):
#     portal = GIS(url, api_key=api_key)
    
#     return portal

def query_gpt(text_list, api_key, max_tokens=30, temperature=0, seed=0, max_try_num=10, tqdm_disable=False, model="gpt-3.5-turbo"):
    openai.api_key = api_key
    result_list = []
    for prompt in tqdm(text_list, disable=tqdm_disable):
        curr_try_num = 0
        while curr_try_num < max_try_num:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role":"user", "content":prompt}],
                    temperature = temperature,
                    max_tokens = max_tokens,
                    top_p = 1,
                    seed = 0,
                    request_timeout=100
                )
                result = response["choices"][0]["message"]["content"]
                result_list.append(result)
                break
            except openai.InvalidRequestError as e:
                print(e)
                return [-1]
            except Exception as e:
                print(e)
                curr_try_num += 1
                if curr_try_num >= max_try_num:
                    result_list.append(-1)
                time.sleep(10)
    return result_list


def query_llama(text_list, api_key, max_tokens=30, temperature=0, seed=0, max_try_num=10, tqdm_disable=False, model="meta-llama/Meta-Llama-3-8B-Instruct"):
#     openai.api_key = api_key

    os.environ["HUGGINGFACE_PROMPT"] = "llama3"
    os.environ["HUGGINGFACE_TOKEN"] = api_key 
    
    result_list = []
    for prompt in tqdm(text_list, disable=tqdm_disable):
#         curr_try_num = 0
#         while curr_try_num < max_try_num:
#             try:

        response = huggingface.ChatCompletion.create(
            model=model,
            messages=[{"role":"user", "content":prompt}],
            temperature = temperature,
            max_tokens = max_tokens,
            top_p = 0.99,
#             top_k = 50,
#             frequency_penalty = 1.0,
#             stream = True
        )
        print(response)
        result = response["choices"][0]["message"]["content"]
        print(result)
        result_list.append(result)
        time.sleep(0.01)
        break
#             except ValidationError as e:
#                 print(repr(exc.errors()[0]['type']))
#                 return [-1]
#             except Exception as e:
#                 print(e)
#                 curr_try_num += 1
#                 if curr_try_num >= max_try_num:
#                     result_list.append(-1)
#                 time.sleep(10)
    return result_list


def fill_in_templates(fill_in_dict, template_str):
    for key, value in fill_in_dict.items():
        if key in template_str:
            template_str = template_str.replace(key, value)
    return template_str   


def get_prompt_for_extracting_modules(question, version='1'):
    with open("./templates/function_desc.txt", "r") as f:
        function_desc = f.read()
        
    if version == '1':
        postfix = ''
    elif version == '2':
        postfix = '_v2'
    else:
        assert(0)
        
    with open(f"./templates/get_modules_template{postfix}.txt", "r") as f:
        template = f.read()

    fill_in_dict = {
        "<MODULE_DESC>": function_desc, 
        "<QUESTION>": question
    }
    template = fill_in_templates(fill_in_dict, template)
    return template


def get_prompt_for_inference_unsupervised(df, index, target_question, coarse_only=False, fine_only=False, random=False, in_context_num=5):        
    template = "<EXAMPLES>\n\n<TARGET>"

    target_df = df.iloc[index]
    candidate_df = df.drop(index)
        
    # choose in-context samples
    scored_df = candidate_df[candidate_df['score'] != -1]
    if len(scored_df) < 1:
        example_desc = ""
    else:
        if len(scored_df) < 5:
            example_df = scored_df
        else:
            if random:
                example_df = scored_df.sample(5)
            else:
                quantile_list = [0, 0.25, 0.5, 0.75, 1]
                selected_df_list = []
                selected_index_list = []
                if fine_only == False or coarse_only:
                    for quant_val in quantile_list:
                        selected_df = scored_df[scored_df['score'] == scored_df['score'].quantile(quant_val, interpolation='lower')]
                        selected_df = selected_df.sample(1)
                        selected_df_list.append(selected_df)
                        selected_index_list.append(selected_df.index[0])

                if coarse_only == False or fine_only:
                    remaining_df = scored_df.drop(selected_index_list, axis=0).reset_index(drop=True)
                    if len(remaining_df) >= 5:
                        extracted_vals = []            
                        for remaining_area_id in remaining_df['area_id']:
                            desc = df[df['area_id'] == remaining_area_id].iloc[0]['desc']
                            extracted_vals.append(np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(desc.split('\n')[1:]))]))
                        extracted_vals = np.stack(extracted_vals)
                        target_val = np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(target_df['desc'].split('\n')[1:]))])

                        scaler = StandardScaler()
                        extracted_vals_normalized = scaler.fit_transform(extracted_vals)
                        target_val_normalized = scaler.transform(target_val.reshape(1, -1))
                        sample_dist = np.linalg.norm(extracted_vals_normalized - target_val_normalized, axis=1)
                        selected_neighbors = sample_dist.argsort()[:3]
                        for neighbor in selected_neighbors:
                            selected_df_list.append(pd.DataFrame(remaining_df.iloc[neighbor]).T)

                if len(selected_df_list) != 0:
                    example_df = pd.concat(selected_df_list, axis=0)
                
        example_desc_list = []
        for example_area_id in example_df['area_id']:
            example_desc = df[df['area_id'] == example_area_id].iloc[0]['desc']
            example_desc_list.append(example_desc)

        example_df['desc'] = example_desc_list    
        example_desc = '\n\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
        
    # Serialize the dataframe
    target_desc = '. '.join(target_df['desc'].split('\n'))
    target_desc += f'.\n{target_question}\nAnswer: '
        
    fill_in_dict = {
        "<EXAMPLES>": example_desc,
        "<TARGET>": target_desc
    }
    template = fill_in_templates(fill_in_dict, template)
    return template



def get_prompt_for_inference_in_context(df, df_record, index, X_train, y_train, X_test, target_question, random=False, coarse_only=False, fine_only=False):        
    template = "<EXAMPLES>\n\n<TARGET>"
                
    # choose in-context samples
    X_train_all = X_train.copy()
    X_train_all['score'] = y_train.to_numpy()
    
    example_df = X_train_all
    candidate_df = df_record.copy()
    
    test_desc = df[df['area_id'] == X_test['area_id'].iloc[index]].iloc[0]['desc']
        
    # choose in-context samples
    scored_df = candidate_df[candidate_df['score'] != -1]  
    if len(scored_df) < 3 and len(scored_df) > 0:
        add_example_df = scored_df
        example_df = pd.concat([example_df, add_example_df], axis=0)
    elif len(scored_df) >= 3:
        if random == False:
            quantile_list = [0.0, 0.5, 1.0]        
            selected_df_list = []
            selected_index_list = []
            if fine_only == False or coarse_only:
                for quant_val in quantile_list:
                    selected_df = scored_df[scored_df['score'] == scored_df['score'].quantile(quant_val, interpolation='lower')]
                    selected_df = selected_df.sample(1)
                    selected_df_list.append(selected_df)
                    selected_index_list.append(selected_df.index[0])

            if coarse_only == False or fine_only:
                remaining_df = scored_df.drop(selected_index_list, axis=0).reset_index(drop=True)
                if len(remaining_df) > 5:
                    extracted_vals = []            
                    for remaining_area_id in remaining_df['area_id']:
                        desc = df[df['area_id'] == remaining_area_id].iloc[0]['desc']
                        extracted_vals.append(np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(desc.split('\n')[1:]))]))
                    extracted_vals = np.stack(extracted_vals)
                    target_val = np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(test_desc.split('\n')[1:]))])

                    scaler = StandardScaler()
                    extracted_vals_normalized = scaler.fit_transform(extracted_vals)
                    target_val_normalized = scaler.transform(target_val.reshape(1, -1))
                    sample_dist = np.linalg.norm(extracted_vals_normalized - target_val_normalized, axis=1)
                    selected_neighbors = sample_dist.argsort()[:3]
                    for neighbor in selected_neighbors:
                        selected_df_list.append(pd.DataFrame(remaining_df.iloc[neighbor]).T)
        
            if len(selected_df_list) != 0:
                add_example_df = pd.concat(selected_df_list, axis=0)
                example_df = pd.concat([example_df, add_example_df], axis=0)

    example_desc_list = []
    for example_area_id in example_df['area_id']:
        example_desc = df[df['area_id'] == example_area_id].iloc[0]['desc']
        print(example_desc)
        example_desc_list.append(example_desc)
    
    example_df['desc'] = example_desc_list    
    example_desc = '\n\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
        
        
    # Serialize the dataframe
    target_desc = '. '.join(test_desc.split('\n'))
    target_desc += f'.\n{target_question}\nAnswer: '
        
    fill_in_dict = {
        "<EXAMPLES>": example_desc,
        "<TARGET>": target_desc
    }
    template = fill_in_templates(fill_in_dict, template)
    return template


def get_prompt_for_inference_transfer(df, df_source, df_record, index, X_test, target_question):        
    template = "<EXAMPLES>\n\n<TARGET>"
    
    scored_df = df_source.copy()
    test_desc = df[df['area_id'] == X_test['area_id'].iloc[index]].iloc[0]['desc']
        
    quantile_list = [0.25, 0.5, 0.75]        
    selected_df_list = []
    selected_index_list = []
    for quant_val in quantile_list:
        selected_df = scored_df[scored_df['score'] == scored_df['score'].quantile(quant_val, interpolation='lower')]
        selected_df = selected_df.sample(1)
        selected_df_list.append(selected_df)
        selected_index_list.append(selected_df.index[0])
    
    example_df = pd.concat(selected_df_list, axis=0)
    example_desc_list = []
    for example_area_id in example_df['area_id']:
        example_desc = df_source[df_source['area_id'] == example_area_id].iloc[0]['desc']
        example_desc_list.append(example_desc)
    
    example_df['desc'] = example_desc_list    
    example_desc1 = '\n\n'.join(['. '.join(df_source[df_source['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])
        
    # choose in-context samples
    candidate_df = df_record.copy()
    scored_df = candidate_df[candidate_df['score'] != -1]        
    if len(scored_df) < 3 and len(scored_df) > 0:
        example_df = scored_df
    elif len(scored_df) >= 3:
        quantile_list = [0.0, 0.5, 1.0]        
        selected_df_list = []
        selected_index_list = []
        for quant_val in quantile_list:
            selected_df = scored_df[scored_df['score'] == scored_df['score'].quantile(quant_val, interpolation='lower')]
            selected_df = selected_df.sample(1)
            selected_df_list.append(selected_df)
            selected_index_list.append(selected_df.index[0])

        remaining_df = scored_df.drop(selected_index_list, axis=0).reset_index(drop=True)
        if len(remaining_df) > 5:
            extracted_vals = []            
            for remaining_area_id in remaining_df['area_id']:
                desc = df[df['area_id'] == remaining_area_id].iloc[0]['desc']
                extracted_vals.append(np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(desc.split('\n')[1:]))]))
            extracted_vals = np.stack(extracted_vals)
            target_val = np.array([float(val) for val in re.findall(r"[-+]?(?:\d*\.*\d+)", '\n'.join(test_desc.split('\n')[1:]))])

            scaler = StandardScaler()
            extracted_vals_normalized = scaler.fit_transform(extracted_vals)
            target_val_normalized = scaler.transform(target_val.reshape(1, -1))
            sample_dist = np.linalg.norm(extracted_vals_normalized - target_val_normalized, axis=1)
            selected_neighbors = sample_dist.argsort()[:2]
            for neighbor in selected_neighbors:
                selected_df_list.append(pd.DataFrame(remaining_df.iloc[neighbor]).T)

        if len(selected_df_list) != 0:
            example_df = pd.concat(selected_df_list, axis=0)

    if len(scored_df) > 0:
        example_desc_list = []
        for example_area_id in example_df['area_id']:
            example_desc = df[df['area_id'] == example_area_id].iloc[0]['desc']
            example_desc_list.append(example_desc)

        example_df['desc'] = example_desc_list    
        example_desc2 = '\n\n'.join(['. '.join(df[df['area_id'] == row['area_id']].iloc[0].desc.split('\n')) + f'.\n{target_question}\nAnswer: ' + str(round(row.score, 3)) for _, row in example_df.iterrows()])

        example_desc1 = f'{example_desc1}\n\n{example_desc2}'
        
    # Serialize the dataframe
    target_desc = '. '.join(test_desc.split('\n'))
    target_desc += f'.\n{target_question}\nAnswer: '
        
    fill_in_dict = {
        "<EXAMPLES>": example_desc1,
        "<TARGET>": target_desc
    }
    template = fill_in_templates(fill_in_dict, template)
    return template


def parse_and_select(answers, version='1'):
    ans_dict = {}
    for answer in answers:
        answer = '1. ' + '1. '.join(answer.split('1. ')[1:]).strip()
        answer = answer.replace('**', '')
        if version == '1':
            parsed_ans = []
            for ans in answer.split('\n'):
                if '):' in ans:
                    parsed_ans.append(ans[3:].split('):')[0].strip() + ')')
                else:
                    parsed_ans.append(ans[3:].strip())                
        elif version == '2':
            parsed_ans = [ans[3:].strip() for ans in answer.split('The total list of selected modules')[1].split('\n')[1:]]
        else:
            assert(0)
                        
        for parsed in parsed_ans:
            if len(parsed) <= 3:
                continue
                
            if 'Loc' not in parsed:
                continue
                
            # Handle exceptional cases here
            if 'get_aggregate_neighbor_info' in parsed:
                parsed = parsed.replace('Func=', '')
            elif 'count_area' in parsed:
                parsed = parsed.replace('Class=', '')
                                
            if parsed not in [*ans_dict]:
                ans_dict[parsed] = 1
            else:
                ans_dict[parsed] += 1
                
    selected_modules = np.array([*ans_dict.keys()])[np.array([*ans_dict.values()]) >= len(answers) // 3]
    return selected_modules

def save_dataset(file_path, info_path, description, answer):
    message_lst = []
    for i in range(len(description)):
        message_lst.append({
            "messages": [
                {"role": "user", "content": description[i]}, 
                {"role": "assistant", "content": str(round(answer[i], 3))}
            ]
        })
        
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in message_lst:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    fileinfo = openai.File.create(
        file=open(file_path, 'rb'),
        purpose='fine-tune'
    )

    with open(info_path, "w") as json_file:
        json.dump(fileinfo, json_file)

    return fileinfo['id']

def random_selection_nightlight(nl_json_path, n_shot, seed):
    #return : dictionary with {'selected_area_id':1, 'not_selected_area_id'}
    np.random.seed(seed)
    with open(nl_json_path,'r') as f:
        data = json.load(f)
    area_id_list = list(data.keys())
    area_id_list.sort(key=lambda area_id: data[area_id]['nightlight']['Nightlight_Average']['val'])
    
    n = int((len(area_id_list) / n_shot))
    n_partition = [area_id_list[i:i+n] for i in range(0, len(area_id_list), n)]
    if len(n_partition) > n_shot:
        n_end = n_partition.pop()
        n_partition[-1]+=n_end
    n_random_selection = [np.random.choice(x) for x in n_partition]
    return n_random_selection