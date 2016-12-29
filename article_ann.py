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
import codecs
import xml.etree.ElementTree as ET
from xlrd import open_workbook
import ann_analysor as aa
import ann_utils as util
import sys


ontologies = 'FMA,GO,HP,PATO'
    # 'FMA,ADO,ICD10,PDON,PATO' # PATO, NIFSTD, DOID
onto_name = {
    'Alzheimer_Ontology': 'http://scai.fraunhofer.de/AlzheimerOntology',
    'FMA': 'http://purl.org/sig/ont/fma/',
    'Human_Disease_Ontology': 'http://purl.obolibrary.org/obo/DOID_',
    'ICD10': 'http://purl.bioontology.org/ontology/ICD10/',
    'Parkinsons_Disease_Ontology': 'http://protegeuserexample#',
    'PATO': 'http://purl.obolibrary.org/obo/PATO_',
    'HPO': 'http://purl.obolibrary.org/obo/HP_',
    'GO': 'http://purl.obolibrary.org/obo/GO_'
}


# entity annotating using NCBO or other annotators later
def annotate_data(txt_file_path, output_folder):
    # onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # for f in onlyfiles:
    #     if f.endswith('.txt'):
            path, fname = os.path.split(txt_file_path)
            fname = fname[:fname.rfind('.')]
            output_file = join(output_folder, fname + '.ncbo')
            obj = None
            if not isfile(output_file):
                print('annotating {} ...'.format(txt_file_path))
                with codecs.open(txt_file_path, encoding='utf-8') as datafile:
                    with codecs.open(output_file, 'w', encoding='utf-8') as outfile:
                        obj = ncboann.annotate(datafile.read(), ontologies)
                        json.dump(
                            obj,
                            outfile
                            )
            else:
                with codecs.open(output_file, encoding='utf-8') as datafile:
                    obj = json.load(datafile)
            # with codecs.open(txt_file_path, encoding='utf-8') as originfile:
            #     convert_NCBO_BRAT(obj, originfile.read(), join(output_folder, fname + '.ann'))


# get the ontology names of the annotated entity
def getEntityType(uri):
    for t in onto_name:
        if uri.startswith(onto_name[t]):
            return t
    return 'Entity'


def count_utf_ascii_len_diff(unicode_str):
    return len(unicode_str.encode('utf-8')) - len(unicode_str)


def convert_bytes_offset_to_utf_offset(unicode_str, byte_offset):
    a_sub = unicode_str.encode('utf-8')[:byte_offset]
    u_sub = ''
    try:
        u_sub = a_sub.decode('utf-8')
    except ValueError:
        print('error', byte_offset, a_sub)
        # u_sub = unicode_str.encode('utf-8')[:byte_offset-1].decode('utf-8')
    return byte_offset - count_utf_ascii_len_diff(u_sub)


def convert_NCBO_BRAT(ncbo_ann, orginal_text, ann_file_path):
    anns = ''
    index = 1
    for ncbo in ncbo_ann:
        entity_type = getEntityType(ncbo['annotatedClass']['@id'])
        if entity_type != 'Entity':
            for loc in ncbo['annotations']:
                anns += (u'T{index}\t{type} {start} {end}\t{text}\n'.format(**{
                    'index': index,
                    'type': entity_type,
                    'start': loc[u'from'] - 1,
                    'end': loc[u'to'],
                    'text': orginal_text[loc['from']-1:loc['to']]
                }))
                anns += (u'#{0}\tAnnotatorNotes T{0}	{1}\n'.format(index, ncbo['annotatedClass']['@id']))
                index += 1
    with codecs.open(ann_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write(anns)


# parse the sapienta annotations
def parse_sepienta(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    sentences = []
    sentence_lst = root.findall(".//s")
    parent_map = {c: p for p in tree.iter() for c in p}
    t2f = {}
    full_text = ''
    for s in sentence_lst:
        so = {'sid': s.attrib['sid'], 'text': s.text}
        if so['text'] is None:
            tlist = s.findall('./text')
            if tlist is not None and len(tlist) > 0:
                so['text'] = "".join([x for t in tlist for x in t.itertext()])
                #' '.join(['' if tnode.text is None else tnode.text for tnode in tlist])
        if so['text'] is not None:
            # get page number
            page = find_page_number(s, parent_map)
            if page is not None:
                so['page'] = page

            # get core concept of the sentence
            sc_list = s.findall('./CoreSc1')
            if sc_list is not None and len(sc_list) > 0:
                so['CoreSc'] = sc_list[0].attrib['type']
            if 'CoreSc' in so:
                t2f[so['CoreSc']] = 1 if so['CoreSc'] not in t2f else 1 + t2f[so['CoreSc']]

            # get the structure label of the sentence
            struct_info = find_sensible_struct_info(s, parent_map)
            if struct_info is not None:
                so['struct'] = struct_info

            # save to full text
            so['start'] = len(full_text)
            so['end'] = so['start'] + len(so['text'])
            full_text += so['text'] + '\n'
            sentences.append(so)
        # else:
        #     print(so['sid'])
    return full_text, sentences


def find_page_number(elem, parent_map):
    if elem in parent_map:
        p = parent_map[elem]
        if 'page' in p.attrib:
            return p.attrib['page']
        else:
            find_page_number(p, parent_map)
    return None


# find the sensible structure information for current sentence
def find_sensible_struct_info(elem, parent_map):
    """
    look for a meaningful section information in current element's ancestors
    Args:
        elem:
        parent_map:

    Returns: the class of the nearest meaningful ancestor

    """
    if elem in parent_map:
        p = parent_map[elem]
        if 'class' in p.attrib \
            and 'unknown' != p.attrib['class'] \
            and 'DoCO:Section' != p.attrib['class'] \
            and 'DoCO:TextChunk' != p.attrib['class']:
            return p.attrib['class']
        else:
            return find_sensible_struct_info(p, parent_map)
    return None


def merge_NCBO_ann(ncbo_file, ann):
    ncbo = None
    with codecs.open(ncbo_file, encoding='utf-8') as read_file:
        ncbo = json.load(read_file)

    for n in ncbo:
        for n_ann in n['annotations']:
            s = binarySearch(ann, n_ann['from'], n_ann['to'])
            if s is not None:
                ano = {'uri': n['annotatedClass']['@id'], 'annotation': n_ann}
                if 'ncbo' in s:
                    s['ncbo'].append(ano)
                else:
                    s['ncbo'] = [ano]
    return ann


def normalise_highlighted_text(ht_text):
    ht_text = ht_text.strip().replace(u'- ', '')
    ht_text = ht_text.replace(u'\ufb01', 'fi').replace(u'\ufb02', 'fl').replace(u'\u2013', '-').replace(u'\u2019', u'’').replace(u'\u2afb ', '').replace(u'\u2afd ', '').replace(u'123 tion', u'tion').replace(u'\ufb00', 'ff').replace(u'\u201c', u'“').replace(u'\u201d', u'”')
        # .replace(u'\xb1', '±').replace(u'\xbc', '¼')
    if ht_text.endswith('-'):
        ht_text = ht_text[:len(ht_text) - 1]
    return ht_text


def merge_highlights(ann, ht):
    matched = 0
    total = 0
    last_matched_page = 1
    for a in ann:
        if 'marked' in a:
            a['marked'] = []
    for page in ht:
        total += len(ht[page])
        for ht_text in ht[page]:
            ht_text = normalise_highlighted_text(ht_text)
            b_matched = False
            for a in ann:
                if True: #'page' in a:
                    # if int(a['page'])>=last_matched_page:
                    if normalise_highlighted_text(a['text']).find(ht_text) >=0:
                        if 'marked' in a:
                            a['marked'].append(ht_text)
                        else:
                            a['marked'] = [ht_text]
                        matched += 1
                        # last_matched_page = int(a['page'])
                        b_matched = True
                        break
            if not b_matched:
                print(page, ht_text)
                    # elif int(a['page'])>int(page):
                    #     # not found in this page
                    #     print(page, ht_text)
                    #     break
    print ('total {0}, mateched {1}'.format(total, matched))


def binarySearch(sents, start, end):
    if len(sents) > 0:
        idx = int(math.ceil(len(sents)/2))
        if sents[idx]['start'] >= end:
            return binarySearch(sents[:idx], start, end)
        elif sents[idx]['end'] < start:
            return binarySearch(sents[idx+1:], start, end)
        elif sents[idx]['start'] <= start and sents[idx]['end'] >= end:
            return sents[idx]
        else:
            print(
                """
                annotation cross sentences...s.start {0}, s.end {1}, a.start {2}, a.end {3}
                sentence {4}
                """
                  .format(sents[idx]['start'], sents[idx]['end'], start, end, sents[idx]))
            return None
    else:
        return None


def read_highlights_json(json_file):
    with codecs.open(json_file, encoding='utf-8') as data_file:
        return json.load(data_file)


def read_highlights(xls_file):
    wb = open_workbook(xls_file)
    sheet = wb.sheets()[0]
    number_of_rows = sheet.nrows
    number_of_columns = 2
    ht = {}
    for row in range(1, number_of_rows):
        ln = sheet.cell(row, 0).value
        txt = sheet.cell(row, 1).value
        txt = txt.replace('\n', ' ')
        if ln in ht:
            ht[ln].append(txt)
        else:
            ht[ln] = [txt]
    return ht


def ann_article(file_path):
    path, file_name = os.path.split(file_path)
    text, sentence = parse_sepienta(os.path.join(path, file_name))

    name = file_name[:file_name.rfind('.')]
    txt_file_path = os.path.join(path, name + '_fulltext.txt')
    with codecs.open(txt_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)
    ann_file_path = os.path.join(path, name + '_ann.json')
    with codecs.open(ann_file_path, 'w', encoding='utf-8') as ann_file:
        json.dump(sentence, ann_file, encoding='utf-8')

    annotate_data(txt_file_path, path)

    ht_file = os.path.join(path, name[:name.rfind('_')] + '_ht.json')
    ann = None
    with codecs.open(ann_file_path, encoding='utf-8') as read_file:
        ann = json.load(read_file)
    ncbo_file = os.path.join(path, name + '_fulltext.ncbo')
    merge_NCBO_ann(ncbo_file, ann)

    if os.path.exists(ht_file):
        ht = read_highlights_json(ht_file)
        merge_highlights(ann, ht)

    with codecs.open(ann_file_path, 'w', encoding='utf-8') as write_file:
        json.dump(ann, write_file)


def test_merge_highlight():
    ht_file = './30-test-papers/11274654_ht.json'
    ann_file = './30-test-papers/11274654_annotated_ann.json'
    ann = util.load_json_data(ann_file)
    for a in ann:
        if 'marked' in a:
            a['marked'] = []
    ht = read_highlights_json(ht_file)
    merge_highlights(ann, ht)
    util.save_json_array(ann, ann_file)


def append_abstract_label(xml_file):
    p, f = os.path.split(xml_file)
    ann_file = os.path.join(p, f[:f.rfind('.')] + '_ann.json')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    abstracts = root.findall(".//abstract")
    if len(abstracts) > 0:
        ab_sents = abstracts[0].findall("s")
        max_ab_sid = int(ab_sents[len(ab_sents)-1].attrib['sid'])
        if max_ab_sid >= 0:
            anns = util.load_json_data(ann_file)
            for ann in anns:
                if int(ann['sid']) <= max_ab_sid:
                    ann['abstract-title'] = True
            util.save_json_array(anns, ann_file)


def append_abstract_label_for_all(xml_path):
    util.multi_thread_process_files(xml_path, 'xml', 10, append_abstract_label)

def main():
    path = './local_exp/42-extra-papers/'
    num_threads = 30
    util.multi_thread_process_files(path, 'xml', num_threads, ann_article)

    # onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # for f in onlyfiles:
    #     if f.endswith('_ann.json'):
    #         aa.analysis_ann(os.path.join(path, f))
    # append_abstract_label_for_all('./20-test-papers/')

if __name__ == "__main__":
    main()

