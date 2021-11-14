#!/usr/bin/env python3
from hn_filter_core import get_stories, filter_stories

from cachier import cachier

n_stories = 20
avg_cluster_size = 3

BOLDON = "\033[1m"
BOLDOFF = "\033[0m"

flatten = lambda n: [k for a in n for k in a]

stories = flatten(get_stories(i+1) for i in range(3))
filtered_stories = filter_stories(stories)

good_stories = filtered_stories['good']
crap_stories = filtered_stories['crap']

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
def read_article(t):
    return [s.replace("[^a-zA-Z]", " ").split(" ") for s in t.split(". ")][:-1]
    
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

@cachier()
def generate_summary(text, top_n=int(n_stories/avg_cluster_size)):
    stop_words = stopwords.words('english')

    # Step 1 - Read text anc split it
    sentences =  read_article(text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True,key=lambda e: e[0])
    return ". ".join(' '.join(e[1]) for e in ranked_sentence[:top_n])

def get_content(url):
    from requests import get
    if not url: return ''
    return get(url, timeout=60).content.decode('utf-8')

@cachier()
def extract_content(url):
    from trafilatura import extract
    from lxml import html
    try:
        content = get_content(url)
        page = html.fromstring(content)
        return content, extract(page, include_comments=False, include_tables=False, no_fallback=True)
    except:
        return '', ''

stories = list()
for story in good_stories:
    print(story['link'])
    try:
        raw, content = extract_content(story['link'])
        density = len(content)/len(raw)
        story['density'] = density
        story['content'] = content
        assert content
        assert density > 0.1
        story['summary'] = generate_summary(story['content'][:1000])
        assert story['summary']
        stories.append(story)
    except Exception as ex:
        print(ex)
        good_stories.remove(story)
        crap_stories.append(story)

# from sklearn.feature_extraction.text import TfidfVectorizer
# import re
# 
# vectorizer = TfidfVectorizer(stop_words={'english'})
# X = vectorizer.fit_transform((story['title'] + ' ' + re.sub('[^A-Za-z]', ' ', story['link']) + ' ' + story['summary'][:30] for story in stories))
# 
# from sklearn.cluster import KMeans
# Sum_of_squared_distances = []
# K = range(2,10)
# 
# for k in K:
   # km = KMeans(n_clusters=k, max_iter=200, n_init=10)
   # km = km.fit(X)
   # Sum_of_squared_distances.append(km.inertia_)
# 
# true_k = 5
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
# model.fit(X)
# labels=model.labels_
# 
# data=[(label,story) for label, story in zip(labels, stories)]
# 
# clusters={}
# for label, story in  data:
    # clusters[label]=clusters.get(label, []) + [story]
# 
# for label, cluster in clusters.items():
    # for story in cluster:
        # print(str(label) + "  " + BOLDON + story['title'] + BOLDOFF)
        # print("  " + story['link'])
        # text = story['summary'].split()
        # print("\n  ", end='')
        # for i, word in enumerate(text[:50]):
            # print(word, end=' ' if (i+1) % 10 else '\n  ')
        # print("")

from nltk.corpus import stopwords
import re

stops = set(stopwords.words('english')) | set([''])
pattern = re.compile('[^a-z]|(^.*ing$)')

import json
def json_hash(*args, **kwargs):
    return json.dumps([args, kwargs])

@cachier(hash_params=json_hash)
def extract_keywords(story):
    word_list = [re.sub(pattern,'', w.lower()) for w in story['content'].split()]
    words = set(word_list) - stops

    counts = {}
    for word in words:
        counts[word] = len([w for w in word_list if w == word])
    return counts


for story in stories:
    story['words'] = extract_keywords(story)

good_stories = stories

# fstories = filter_stories(stories)
# good_stories = fstories['good']
# crap_stories += fstories['crap']

for story in good_stories[:n_stories][::]:
    try:
        words = story['words'].keys()
        keywords = set(words)
        for s in stories:
            if s != story:
                keywords -= set(s['words'].keys())
        kw = [k for k,v in sorted(story['words'].items(), reverse=True, key=lambda p: p[1])[:8]]
        print(BOLDON + story['title'] + BOLDOFF)
        print("  ", kw)
        text = story['summary'].split()
        print("\n  ", end='')
        for i, word in enumerate(text[:50]):
            print(word, end=' ' if (i+1) % 10 else '\n  ')
        print("\n  " + story['link'])     
        print("\n  Density: ", story['density'])
    except:
        pass

print("Good: ", len(good_stories), sum(story['density'] for story in good_stories)/len(good_stories))
densities = [story['density'] for story in crap_stories if 'density' in story.keys()]
print("Crap: ", len(crap_stories), sum(densities)/len(densities))
