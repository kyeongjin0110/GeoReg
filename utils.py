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
from arcgis.gis import GIS
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


def login_gis_portal(
    api_key,
    url="https://www.arcgis.com"
):
    portal = GIS(url, api_key=api_key)
    
    return portal


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
                    top_p = 1.0,
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