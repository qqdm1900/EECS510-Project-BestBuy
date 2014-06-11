#!/usr/bin/env python
"""
Group members: Mengyun Lv & Yuetong Zhao
course project for EECS 510
Usage: (1) put train and test csv data files inside the data folder
       (2) run "./clean_data.py" 
       (3) run "./train_predict.py ../data/clean_train.csv ../data/clean_test.csv"
       (4) the resultant prediction file will be generated in the data folder
"""

from datetime import datetime
import os, sys
from nltk.stem import WordNetLemmatizer
from misc import readfile, writefile, get_words
data_path = '../Data' 
train_file = os.path.join(data_path, 'train.csv')
test_file = os.path.join(data_path, 'test.csv')
new_train_file = os.path.join(data_path, 'clean_train.csv')
new_test_file = os.path.join(data_path, 'clean_test.csv')

st_date = datetime.strptime('2011-08-11 04:00:17', '%Y-%m-%d %H:%M:%S')
_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not _root_dir in sys.path:
    sys.path.insert(0, _root_dir)

def word_split(word):
    if word[-1].isalpha():
        return word
    if word.isdigit():
        return word
    for idx in xrange(len(word) - 1, 0, -1):
        if not (word[idx].isdigit() and word[idx - 1].isdigit()):
            break
    if idx <= 1:
        return word
    return word[:idx], word[idx:]

def get_new_time(t):
    try:
        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
    except:
        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    new_time = str((t - st_date).days)
    return new_time 

def clean_query(raw_query, lemmatizer, local_cache):
    raw_query = raw_query.lower().strip()
    if raw_query in local_cache:
        return local_cache[raw_query]
    words = get_words(raw_query)
    new_words = list()
    for w in words:
        if not w.isdigit() and not w.isalpha():
            split = word_split(w)
        else:
            split = w
        if type(split) == type(()):
            new_words.extend(list(split))
        else:
            new_words.append(split)
    new_query = ''
    for w in new_words:
        lemma = lemmatizer.lemmatize(w)
        if not lemma.isdigit() and not lemma.isalpha() and len(lemma) >= 4:
            split = word_split(w)
            if type(split) == type(()):
                w, num = split
                lemma = ' '.join([w, num])
        new_query += lemma + ' '
    new_query = new_query[0:-1]
    local_cache[raw_query] = new_query
    return new_query

def fix_query(in_file, out_file, file_type):
    local_cache = dict()
    lemmatizer = WordNetLemmatizer()
    reader = readfile(in_file)
    with open(out_file, 'w') as writer:
        writer.write('data:\n')
        if file_type == 'train':
            for (user, sku, category, raw_query, click_time, __query_time) in reader:
                new_query = clean_query(raw_query, lemmatizer, local_cache)
                new_click_time = get_new_time(click_time) 
                outline = ','.join([user, sku, category, new_query, new_click_time])
                writer.write(outline + '\n')
        elif file_type == 'test':
            for (user, category, raw_query, click_time, __query_time) in reader:
                new_query = clean_query(raw_query, lemmatizer, local_cache)
                new_click_time = get_new_time(click_time)
                outline = ','.join([user, category, new_query, new_click_time])
                writer.write(outline + '\n')
        else:
            raise Exception('Query Correction Failed!')

if __name__ == '__main__':
    fix_query(train_file, new_train_file, 'train')
    print 'train data cleaned!'
    fix_query(test_file, new_test_file, 'test')
    print 'test data cleaned!'
