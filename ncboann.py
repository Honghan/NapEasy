import urllib2
import requests
import httplib2
import json
import os
from pprint import pprint
import ann_utils as utils
import codecs

REST_URL = "http://data.bioontology.org"
# Honghan's api key
API_KEY = "5db4a03d-144f-4903-9933-aaf326dd7786"
ontologies = 'BTO,DRON,NDDF'


# create the url for http get
def construct_url(text, ontologies):
    u = REST_URL + "/annotator?text=" + urllib2.quote(text)
    op = ''
    for o in ontologies:
        op += o + ","
    if len(op) > 0:
        op = op[:len(op)-1]

    u = u + "&ontologies=" + op
    return u


# create the data object (dictionary) for http post
def construct_postobj(text, ontos):
    return {'text': text, 'ontologies': ontos, 'apikey': API_KEY}


# post to get annotation
def post_json(postobj):
    r = requests.post(REST_URL + "/annotator", data=postobj)
    response = r.content
    return json.loads(response)


# httpget to get annotation
def get_json(url):
    opener = urllib2.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    return json.loads(opener.open(url).read())


# annotate a text
def annotate(text, ontos = None):
    ann_ontos = ontos
    if ontos is None:
        ann_ontos = ontologies
    return post_json(construct_postobj(text, ann_ontos))


# test
def test():
    text_to_annotate = "12 While oral corticosteroids are the cornerstone of management of acute, moderate or severe asthma,6 several reports have recently shaken the belief that they are equally effective for all patients with asthma, showing that children with viral-induced wheezing21 and smoking adults22 are corticosteroid-resistant."
    # Annotate using the provided text
    annotations = annotate(text_to_annotate)
    print(json.dumps(annotations))
    # annotations = get_json(construct_url(text_to_annotate, ontologies))


def match_concepts(text, concepts):
    t = text.lower()
    ret = []
    for c in concepts:
        p = t.find(c)
        if p >= 0:
            ret.append([c, p])
    return ret


def file_match_concepts(ann_file, concepts):
    anns = utils.load_json_data(ann_file)
    for ann in anns:
        ret = match_concepts(ann['text'], concepts)
        if len(ret) > 0:
            print ret, ann['sid']


def load_brain_regions(f):
    concepts = []
    with codecs.open(f, encoding='utf-8') as rf:
        concepts = [r.split('\t')[1].replace('\n', '') for r in rf.readlines()]
    return concepts


if __name__ == "__main__":
    concepts = load_brain_regions('./resources/brain-regions-without-fma.tsv')
    file_match_concepts('./anns_v2/Chechko et al., (2014) - Neural correlates of unsuccessful memory performance in MCI_annotated_ann.json', concepts)