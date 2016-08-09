#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ncboann
import json
import nltk
from nltk.data import load
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import os
import math


ontologies = 'FMA,DOID,ADO,ICD10,PDON' # PATO, NIFSTD
onto_name = {
    'Alzheimer_Ontology': 'http://scai.fraunhofer.de/AlzheimerOntology',
    'FMA': 'http://purl.org/sig/ont/fma/',
    'Human_Disease_Ontology': 'http://purl.obolibrary.org/obo/DOID_',
    'ICD10': 'http://purl.bioontology.org/ontology/ICD10/',
    'Parkinsons_Disease_Ontology': 'http://protegeuserexample#'
}
article_path = "./data/"
output_folder = "./data/"


def annotate_data(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for f in onlyfiles:
        if f.endswith('.txt'):
            fname = f[:f.rfind('.')]
            output_file = join(output_folder, fname + '.ncbo')
            obj = None
            if not isfile(output_file):
                print('annotating {} ...'.format(f))
                with open(join(path, f)) as datafile:
                    with open(output_file, 'w') as outfile:
                        obj = ncboann.annotate(datafile.read(), ontologies)
                        json.dump(
                            obj,
                            outfile
                            )
            else:
                with open(output_file) as datafile:
                    obj = json.load(datafile)
            with open(join(path, f)) as originfile:
                convert_NCBO_BRAT(obj, originfile.read(), join(output_folder, fname + '.ann'))


def getEntityType(uri):
    for t in onto_name:
        if uri.startswith(onto_name[t]):
            return t
    return 'Entity'


def convert_NCBO_BRAT(ncbo_ann, orginal_text, ann_file_path):
    anns = ''
    index = 1
    for ncbo in ncbo_ann:
        entity_type = getEntityType(ncbo['annotatedClass']['@id'])
        for loc in ncbo['annotations']:
            anns += ('T{index}\t{type} {start} {end}\t{text}\n'.format(**{
                'index': index,
                'type': entity_type,
                'start': loc[u'from'] - 1,
                'end': loc[u'to'],
                'text': orginal_text[loc['from']-1:loc['to']]
            }))
            anns += ('#{0}\tAnnotatorNotes T{0}	{1}\n'.format(index, ncbo['annotatedClass']['@id']))
            index += 1
    with open(ann_file_path, 'w') as outfile:
        outfile.write(anns)

annotate_data(article_path)
print('all done')
