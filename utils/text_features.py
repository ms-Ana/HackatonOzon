from scipy.spatial.distance import cosine, euclidean
import numpy as np
import nltk
import re
import json
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

def text_dense_distances(ozon_embedding:np.ndarray, 
                         comp_embedding:np.ndarray):
    """Calculate Euclidean and Cosine distances between
    ozon_embedding and comp_embedding.
    """
    pair_features = []
    if ozon_embedding is None or comp_embedding is None:
        pair_features = [-1, -1]
    elif len(ozon_embedding) == 0 or len(comp_embedding) == 0:
        pair_features = [-1, -1]
    else:
        pair_features.append(
            euclidean(ozon_embedding, comp_embedding)
        )
        cosine_value = cosine(ozon_embedding, comp_embedding)
        
        pair_features.append(cosine_value)

    return pair_features

def change_dimensions(text: str):
    dimensions_change = {
        'cm ':'см', 'mm ':'мм', 'm ':'м', 'dm ':'дм',
        'A ':'А', 'Ампер ':'А', 'Ам ':'А', 'W ':'Вт ', 'V ':'В', 'Watt ':'Вт', 'Вольт ':'В', 'Ohm ':'Ом',
        'Gb ':'Гб', 'ГБ ':'Гб', 'Tb ':'Тб ', 'ТБ ':'Тб', 'Mb ':'Мб', 'МБ ':'Мб', 'Kb ':'Кб', 'КБ ':'Кб', 
        'byte ':'б', 'b ':'б', 'gb ':'Гб', 'kb ':'Кб', 'mb ':'Мб', 'tb ':'Тб',
        '"':'дюйм', 'inch ':'дюйм', 'inches ':'дюйм',
        'kg ':'кг', 'g ':'г', 'mg ':'мг',
        'hHz ':'гГц', 'hhz ':'гГц', 'gHz ':'ГГц', 'ghz ':'ГГц', 'kHz ':'кГц', 'khz ':'кГц', 'mHz ':'мГц', 'mhz ':'мГц', 'Hz ':'Гц', 'hz ':'Гц',
        'ч ':'час', 'min ':'минут', 'мин ':'минут', 'sec ':'секунд', 'сек ':'секунд', 'ms ':'милисекунд',
        'милисек ':'милисекунд', 'years ':'год', 'year ':'год', 'year ':'год', 'months ':'месяц', 'month ':'месяц'
    }
    text = re.sub(r'[,]',' ',text)
    for pattern, replacement in dimensions_change.items():
        text = re.sub(pattern, replacement, text)
    return text

def find_dimensions(text):
    dimensions_regex = r'\b(\d+(\.\d+)?)\s?(шт|см|мм|м|дм|А|Вт|В|Ом|Гб|Тб|Мб|Кб|байт|дюйм|кг|г|мг|гГц|ГГц|кГц|мГц|Гц|час|минут|секунд|милисекунд|год|месяц)\b'
    matches = re.findall(dimensions_regex, text)
    dimensions = [match[0] + ' ' + match[2] for match in matches]
    return dimensions

def del_dimensions(text):
    dimensions_regex = r'\b(\d+(\.\d+)?)\s?(шт|см|мм|м|дм|А|Вт|В|Ом|Гб|Тб|Мб|Кб|байт|дюйм|кг|г|мг|гГц|ГГц|кГц|мГц|Гц|час|минут|секунд|милисекунд|год|месяц)\b'
    text = re.sub(dimensions_regex, '', text)
    return text

def find_code(text):
    pattern = r'\b([A-Za-z0-9]*-*[A-Za-z0-9]+-[A-Za-z0-9]+)\b'
    code = re.findall(pattern, text)
    return code

def del_code(text):
    pattern = r'\b([A-Za-z0-9]*-*[A-Za-z0-9]+-[A-Za-z0-9]+)\b'
    text = re.sub(pattern, '', text)
    return text

def change_countr(text):
    countr_change = {
        'usd':'USD',
        'Usd':'USD',
        'eur':'USD',
        'Eur':'EUR',
        'Euro':'EUR',
        'EURO':'EUR',
        'euro':'EUR',
        'Uk':'UK',
        'uk':'UK',
        'Rus':'RUS',
        'rus':'RUS'
    }
    
    for pattern, replacement in countr_change.items():
        text = re.sub(pattern, replacement, text)
    return text

def find_countr(text):
    rule = r"\b(?:EUR|USD|RUS|UK|Китай|Япония)\b"
    countr = re.findall(rule, text)
    return countr

def find_rus_abbreviations(text):
    list_abr = []
    abbreviations = re.findall(r'\b[А-Я]{2,}\b', text)
    if abbreviations is not None:
        for abr in abbreviations:
            list_abr.append(str(abr))
    return list_abr
    

def find_eng_abbreviations(text):
    list_abr = []
    abbreviations = re.findall(r'\b[A-Z]{2,}\b', text)
    if abbreviations is not None:
        for abr in abbreviations:
            list_abr.append(str(abr))
    return list_abr

def find_model(text):
    rule = r'\b(?=\w+\d)(?=\w*[a-zA-Zа-яА-Я])\w+[-]*\b(?<!-)'
    model = re.findall(rule, text)
    return model

def define_brands(etl: pd.DataFrame):
    brands_list = []

    for item in etl["characteristic_attributes_mapping"]:
        if item is not None:
            attributes = json.loads(item)
        if "Бренд" in attributes and attributes["Бренд"] not in brands_list:
            brand = attributes["Бренд"]
            brands_list.append(brand)
    
    brands_list = [re.sub(r'\[|\]|\'', '', str(item)) for item in brands_list]
    
    return brands_list

def define_colors(etl: pd.DataFrame):
    color_list = []
    for item in etl["characteristic_attributes_mapping"]:
        if item is not None:
            attributes = json.loads(item)
        if "Цвет товара" in attributes:
            colors = attributes["Цвет товара"]
            for color in colors:
                if color not in color_list:
                    color_list.append(color)
    color_list = [re.sub(r'\[|\]|\'', '', str(item)) for item in color_list]
    
    return color_list

def text_preprocessing(text):
    if text is None:
        return None
    ser = []
    num_del_rule = r'\b(?:\d+\w*|\w*\d+)\w*\b'
    spec_del_rule = r'[^a-zA-Zа-яА-Я0-9\s]'
    del_punkt_rule = r'[^\w\s]'
    
    text = text.lower() # приведение к нижнему регистру
    
    text = re.sub(num_del_rule, '', text) # удаление всех чисел
    text = re.sub(spec_del_rule, '', text) # удаление спец символов
    
    stop_words = ['для', 'в', 'с', 'под', 'и', 'или', 'на'] + list(stopwords.words('russian'))
    text = re.sub(del_punkt_rule, '', text) # удаление пунктуации

    tokens = nltk.word_tokenize(text) # токенизация 
    
    lemmatizer = WordNetLemmatizer() # лемматизируем текст
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words] # удаляем стоп-слова
    tokens = [word for word in tokens if len(word) > 1] # удаляем слова короче 2х символов
    
    
    preprocessed_text = ' '.join(tokens) # собираем обработанный текст обратно
    ser.append(preprocessed_text)
    ser.append(tokens)
    
    return ser

def create_tokens(text):
    if text is None:
        return None
    ser = []
    del_punkt_rule = r'[^\w\s]'
    
    text = text.lower() # приведение к нижнему регистру
    
    stop_words = ['для', 'в', 'с', 'под', 'и', 'или', 'на'] + list(stopwords.words('russian'))
    text = re.sub(del_punkt_rule, '', text) # удаление пунктуации

    tokens = nltk.word_tokenize(text) # токенизация 
    
    lemmatizer = WordNetLemmatizer() # лемматизируем текст
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words] # удаляем стоп-слова
    tokens = [word for word in tokens if len(word) > 1] # удаляем слова короче 2х символов
    
    ser.append(tokens)
    
    return ser

def jaccard_similarity(list1, list2):
    if list1 is None or list2 is None:
        return 0
    if len(list1) == 0 and len(list2) == 0:
        return 1
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union != 0:
        return float(intersection) / union
    else:
        return 0
    
def abr_dot(text):
    return re.sub(r'\b\w+\.', '', text)

def create_text_features(features: pd.DataFrame) -> pd.DataFrame:
    features["name_preprocessed"] = features.apply(lambda row: change_dimensions(row['name']), axis=1)
    features["name_dimensions"] = features.apply(lambda row: find_dimensions(row['name_preprocessed']), axis=1)
    features["name_preprocessed"] = features.apply(lambda row: del_dimensions(row["name_preprocessed"]), axis=1) 
    features["name_code"] = features.apply(lambda row: find_code(row['name_preprocessed']), axis=1)
    features['name_preprocessed'] = features.apply(lambda row: del_code(row['name_preprocessed']), axis=1)  
    features["name_preprocessed"] = features.apply(lambda row: change_countr(row['name_preprocessed']), axis=1)
    features["name_rus_abbreviations"] = features.apply(lambda row: find_rus_abbreviations(row['name_preprocessed']), axis=1)
    features["name_eng_abbreviations"] = features.apply(lambda row: find_eng_abbreviations(row['name_preprocessed']), axis=1)
    features["name_model"] = features.apply(lambda row: find_model(row['name_preprocessed']), axis=1)
    
    features[["tokens"]]= (
        features[["name"]].apply(
            lambda x: pd.Series(create_tokens(*x)), axis=1
        )
    )

    features[["name_full_preprocessed", "name_tokens"]] = (
        features[["name"]].apply(
            lambda x: pd.Series(text_preprocessing(*x)), axis=1
        )
    )
    return features
