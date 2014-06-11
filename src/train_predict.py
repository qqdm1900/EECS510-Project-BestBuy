#!/usr/bin/env python
"""
Group members: Mengyun Lv & Yuetong Zhao
course project for EECS 510
Usage: (1) put train and test csv data files inside the data folder
       (2) run "src/clean_data.py" 
       (3) run "./train_predict.py ../data/clean_train.csv ../data/clean_test.csv"
       (4) the resultant prediction file will be generated in the data folder
"""

import time
from datetime import datetime
import multiprocessing
import os, sys
from collections import defaultdict
from misc import readfile, writefile, get_words

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# file locations and time
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)
clicked_map = dict()
data_path = '../Data'
out_buffer_path = '../buffer'
predict_file = os.path.join(data_path, 'prediction.csv')
new_train_file = sys.argv[1]
new_test_file = sys.argv[2]

st_date = datetime.strptime('2011-08-11 04:00:17', '%Y-%m-%d %H:%M:%S')
ed_date = datetime.strptime('2011-10-31 10:17:42', '%Y-%m-%d %H:%M:%S')

# parameters for the algorithm
magic_num = 100000
proc_num = 2

'time parameters'
duration = (ed_date - st_date).days
block_size = 12
block = duration / block_size
MAX_BLOCK = block_size - 1

'unigram and bigram parameters'
GLOBAL_QUERY = 6
GLOBAL_BIGRAM_QUERY = 6
w1 = 0.7
w2 = 0.3

SUM = 'SUM'
SUM_SIZE = 'SUM_SIZE'
PREDICT_HOT_SIZE = 'PREDICT_HOT_SIZE'
HOT = 'HOT'
BIGRAM_HOT = 'BIGRAM_HOT'
HOT_SIZE = 'HOT_SIZE'

MAX_TEST_LINE = 2000000 # do prediction for this number of lines 
STEP_SIZE = MAX_TEST_LINE / (100 * proc_num) 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ++++++++++++++++++++++++ GENERAL FUNCTIONS  +++++++++++++++++++++++++++++
def init():
    with open(new_train_file) as fr:
        raw_data = fr.readlines()
    raw_data = raw_data[1:]
    for line in raw_data:
        user, sku, __category, raw_query, __click_time = line.split(',')
        if user not in clicked_map:
            clicked_map[user] = dict()
        if raw_query not in clicked_map[user]:
            clicked_map[user][raw_query] = list()
        clicked_map[user][raw_query].append(sku)

def rank_predictions(guesses, user, query):
    if user in clicked_map and query in clicked_map[user]:
        clicked_skus = clicked_map[user][query]
        clicked = []
        res = []
        for sku in guesses:
            if sku in clicked_skus:
                clicked.append(sku)
            else:
                res.append(sku)
        res.extend(clicked)
        return res
    else:
        return guesses

def bigram_word(raw_query, freq_words, cat):
    words = get_words(raw_query)
    words = [w for w in words if w in freq_words[cat]].sort()
    bigram = [str(words[i]) + '_' + str(words[j]) for i in range(len(words)) for j in range(i + 1, len(words))]
    return bigram

# +++++++++++++++++++++++++  TRAIN ++++++++++++++++++++++++++++++++++++++++++
def count_items():
    f_in = readfile(new_train_file)
    item_count = defaultdict(lambda: defaultdict(int))
    time_item_count = defaultdict(lambda:defaultdict(lambda: defaultdict(int)))
    index = 0
    for (__user, sku, category, __query, click_time) in f_in:
        time_block = min(int(click_time) / block, MAX_BLOCK)
        index += 1
        item_count[category][sku] += magic_num
        time_item_count[time_block][category][sku] += magic_num
    item_sort = dict()
    for category in item_count:
        item_sort[category] = sorted(item_count[category].items(), key=lambda x: x[1], reverse=True)
    smooth_time_item_count = defaultdict(lambda:defaultdict(lambda: defaultdict(int)))
    for time_block in time_item_count:
        for cat in time_item_count[time_block]:
            for sku in time_item_count[time_block][cat]:
                smooth_time_item_count[time_block][cat][sku] = item_count[cat][sku] * 3.0 / block_size
    for time_block in time_item_count:
        for cat in time_item_count[time_block]:
            for sku in time_item_count[time_block][cat]:
                smooth_time_item_count[time_block][cat][sku] = time_item_count[time_block][cat][sku]
                if time_block == 0 or time_block == MAX_BLOCK:
                    smooth_time_item_count[time_block][cat][sku] += time_item_count[time_block][cat][sku]
                if time_block >= 1:
                    smooth_time_item_count[time_block][cat][sku] += time_item_count[time_block - 1][cat][sku]
                if time_block < MAX_BLOCK:
                    smooth_time_item_count[time_block][cat][sku] += time_item_count[time_block + 1][cat][sku]
    return item_count, item_sort, smooth_time_item_count
    
def catgory_stat(item_sort, time_item_count):
    cat_count = defaultdict(lambda: defaultdict(int))
    for cat in item_sort:
        freq_queries = 0
        index = 0
        __jdx = 0
        sum_query = sum([i[1] for i in item_sort[cat]])
        sum_size = len(item_sort[cat])
        while True:
            if index >= sum_size or item_sort[cat][index][1] < GLOBAL_QUERY:
                break
            freq_queries += item_sort[cat][index][1]
            index += 1
        if index < 5:
            index = 5 
        cat_count[cat][HOT_SIZE] = index
        cat_count[cat][SUM_SIZE] = sum_size
        cat_count[cat][HOT] = freq_queries
        cat_count[cat][SUM] = sum_query
        cat_count[cat][PREDICT_HOT_SIZE] = index
        print 'freq index = %d' % index
    for t in range(block_size):
        for cat in item_sort:
            cat_count[cat][t] = cat_count[cat][SUM]
    return cat_count    

def unigram_model(item_sort, cat_count):
    f_in = readfile(new_train_file)
    item_word = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    cat_word = defaultdict(lambda: defaultdict(int))
    index = 0
    for (__user, sku, category, raw_query, ___click_time) in f_in:
        index += 1
        bound = cat_count[category][HOT_SIZE]
        popular = [i[0] for i in item_sort[category][0:bound]]
        if sku in popular:
            words = get_words(raw_query)
            for w in words:
                item_word[category][sku][w] += magic_num
                cat_word[category][w] += magic_num
    return item_word, cat_word

def bigram_model(item_word, item_sort, cat_count):
    freq_sku_words = defaultdict(lambda: defaultdict(set))
    for cat in item_word:
        for sku in item_word[cat]:
            hots = item_word[cat][sku].items()
            freq_sku_words[cat][sku] = set([i[0] for i in hots if i[1] >= GLOBAL_BIGRAM_QUERY])

    freq_words = dict() 
    for cat in freq_sku_words:
        freq_words[cat] = set()
        for sku in freq_sku_words[cat]:
            freq_words[cat] = freq_words[cat].union(freq_sku_words[cat][sku])

    f_in = readfile(new_train_file)
    bigram_item_word = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    index = 0
    for (__user, sku, category, raw_query, ___click_time) in f_in:
        index += 1
        bound = cat_count[category][HOT_SIZE]
        popular = [i[0] for i in item_sort[category][0:bound]]
        if sku in popular:
            bigram = bigram_word(raw_query, freq_words, category)
            for w in bigram:
                bigram_item_word[category][sku][w] += magic_num
                cat_count[category][BIGRAM_HOT] += magic_num
    return bigram_item_word, cat_count, freq_words

# +++++++++++++++++++++++++++++++++++  PREDICTION +++++++++++++++++++++++++++++++
# three models to switch between for queries of different frequcies.
def naive_bayes_query_prediction(words, cat, sku, alpha, beta, item_word, item_count, cat_count):
    cat_c = cat_count[cat][HOT]
    p_i = (item_count[cat][sku] + alpha) * 1.0 / (cat_c + beta)
    p = p_i
    for w in words:
        p_wi = (item_word[cat][sku].get(w, 0) + alpha) * 1.0 / (cat_c + beta)
        p *= p_wi / p_i
    return p 

def bayes_bigram_predict(bigram, cat, sku, alpha, beta, bigram_item_word, item_count, cat_count, month_cat_item_dict, t):
    cat_m = cat_count[cat][t]
    p_m = (month_cat_item_dict[t][cat][sku] + alpha) * 1.0 / (cat_m + beta)
    cat_c = cat_count[cat][HOT]
    cat_bigram_c = cat_count[cat][BIGRAM_HOT]
    p_i = (item_count[cat][sku] + alpha) * 1.0 / (cat_c + beta)
    p = p_m
    for w in bigram:
        p_wi = (bigram_item_word[cat][sku].get(w, 0) + alpha) * 1.0 / (cat_bigram_c + beta)
        p *= p_wi / p_i
    return p 

def bayes_query_predict(words, cat, sku, alpha, beta, item_word, item_count, cat_count, month_cat_item_dict, t):
    cat_m = cat_count[cat][t]
    cat_c = cat_count[cat][HOT]
    p_m = (month_cat_item_dict[t][cat][sku] + alpha) * 1.0 / (cat_m + beta)
    p_i = (item_count[cat][sku] + alpha) * 1.0 / (cat_c + beta) 
    p = p_m
    for w in words:
        p_wi = (item_word[cat][sku].get(w, 0) + alpha) * 1.0 / (cat_c + beta)
        p *= p_wi / p_i
    return p 

def boosting_bayes(bigram, words, cat, sku, alpha, beta, item_word, bigram_item_word, item_count, cat_count, month_cat_item_dict, t):
    p1 = bayes_query_predict(words, cat, sku, alpha, beta, item_word, item_count, cat_count, month_cat_item_dict, t)
    p2 = bayes_bigram_predict(bigram, cat, sku, alpha, beta, bigram_item_word, item_count, cat_count, month_cat_item_dict, t)
    return w1 * p1 + w2 * p2

def make_predictions(st_line, ed_line, predict_file, pname, models):
    cat_count, item_count, item_sort, alpha, beta, item_word, bigram_item_word, time_cat_item_dict, cat_word, freq_words = models[0]
    f_in = readfile(new_test_file)
    f_out = writefile(predict_file)
    line_index = 0
    for (user, category, raw_query, click_time) in f_in:
        line_index += 1
        if line_index < st_line:
            continue
        if line_index > ed_line:
            break
        if line_index % STEP_SIZE == 0:
            print '%s--%d' % (pname, line_index / STEP_SIZE)
        time_block = min(int(click_time) / block, MAX_BLOCK)
        try:
            bound = cat_count[category][PREDICT_HOT_SIZE]
            hots = [x[0] for x in item_sort[category][0:bound]]
        except:
            f_out.writerow(["0"])
            continue
        try:
            bigram = bigram_word(raw_query, freq_words, category)
            words = get_words(raw_query)
            query_size = sum([cat_word[category][w] for w in words])
            if query_size >= 100 and len(bigram) > 0:
                "queries that are frequent enough and can generate bigram features can be predicted by boosting model"
                rank = [[sku, boosting_bayes(bigram, words, category, sku, alpha, beta, item_word, bigram_item_word, item_count, cat_count, time_cat_item_dict, time_block)] for sku in hots]
            elif query_size >= 100 and len(bigram) == 0:
                "if hot enough but can not generate bigram features then use naive bayes with time information"
                rank = [[sku, bayes_query_predict(words, category, sku, alpha, beta, item_word, item_count, cat_count, time_cat_item_dict, time_block)] for sku in hots]
            else:
                "otherwise use naive bayes"
                rank = [[sku, naive_bayes_query_prediction(words, category, sku, alpha, beta, item_word, item_count, cat_count)] for sku in hots]
            rank = sorted(rank, key=lambda x:x[1], reverse=True)
            guesses = [i[0] for i in rank[0:5]]
            guesses = rank_predictions(guesses, user, raw_query)

            f_out.writerow([" ".join(guesses)])
        except (TypeError, KeyError):
            f_out.writerow([" ".join(hots[0:5])])

def gen_output(outfile, inbasepath, file_name, proc_num):
    parameters = mulproc_vars(inbasepath, file_name, proc_num)
    with open(outfile, 'w') as  fw:
        fw.write('sku\n')
        for index in range(proc_num):
            filename, __st, __ed, __pname = parameters[index]
            with open(filename) as fr:
                line = fr.readline()
                while True:
                    line = fr.readline()
                    if not line:
                        break
                    fw.write(line[0:-2] + '\n')

def mulproc_vars(buffer_path, file_name, proc_num):
    block = int(MAX_TEST_LINE / proc_num)
    ret = list()
    for index in range(proc_num):
        filename = os.path.join(buffer_path, '%s_%d.csv' % (file_name, index)) 
        pname = '--p_%d' % index
        st = index * block + 1
        ed = (index + 1) * block if index != (proc_num - 1) else MAX_TEST_LINE
        ret.append([filename, st, ed, pname])
    return ret

def main():
    parameters = mulproc_vars(out_buffer_path, 'buffer', proc_num)
    init()
    st_time = time.time()
    models = list()
    item_count, item_sort, month_item_count = count_items()
    cat_count = catgory_stat(item_sort, month_item_count)
    item_word, cat_word = unigram_model(item_sort, cat_count)
    bigram_item_word, cat_count, freq_words = bigram_model(item_word, item_sort, cat_count)
    models.append([cat_count, item_count, item_sort, 1, 100, item_word, bigram_item_word, month_item_count, cat_word, freq_words])
    ed_time = time.time()
    print '%f time spent on training' % (ed_time - st_time)

    st_time = time.time()
    pool = list()
    print 'number of processors = %d' % (proc_num)
    for pdx in range(proc_num):
        filename, st, ed, pname = parameters[pdx]
        p = multiprocessing.Process(target=make_predictions, args=(st, ed, filename, pname, models))
        p.start()
        pool.append(p)
    for pdx in range(proc_num):
        p = pool[pdx]
        p.join()
    ed_time = time.time()
    print '%f time spent on prediction' % (ed_time - st_time)

    gen_output(predict_file, out_buffer_path, 'buffer', proc_num)

if __name__ == '__main__':
    main()
    print 'Done!'
