#!/usr/bin/env python
"""
This file contains small functions that are shared by the main.py and clean_data.py
"""
import string
import csv

def get_words(raw):
    raw.strip()
    raw = raw.lower()
    res = ''
    for ch in raw:
        res += ch if ch not in set(string.punctuation) else ' '
    res = res.split(' ')
    words = [w for w in res if is_legal(w)]
    return words

def is_legal(word):
    breakers = set(["a", "an", "and", "are", "as", "at", "be", "but", "by",
          "for", "if", "in", "into", "is", "it",
          "no", "not", "of", "on", "or", "such",
          "that", "the", "their", "then", "there", "these",
          "they", "this", "to", "was", "will", "with"])
    if word.find(' ') > 0 or len(word) <= 0 or word in breakers:
        return False
    else:
        return True

def readfile(f):
    infile = open(f)
    f_in = csv.reader(infile, delimiter=",")
    f_in.next() 
    return f_in

def writefile(f):
    outfile = open(f, 'w')
    f_out = csv.writer(outfile, delimiter=",")
    f_out.writerow(["sku"])
    return f_out
