#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import codecs
import article_ann as pann
import matplotlib.pyplot as plt
import random
import os
import re
import nltk
from nltk.parse.stanford import StanfordParser
import ann_utils as utils
import threading

# the lock for gain access to the shared variable
thread_lock = threading.Lock()

# stanford model file path
stanford_language_model_file = "/Users/jackey.wu/Documents/working/libraries/" \
                   "stanford-english-corenlp-2016-01-10-models/" \
                   "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"

# pattern output folder
pattern_output_folder = './training/patterns/'


class SubjectPredicate:
    def __init__(self, sub, pred):
        self.p = pred
        self.s = sub

    def predicate(self):
        return self.p

    def subject(self):
        return self.s

    def pred_str(self):
        return u' '.join([] if self.p is None else self.p)

    def sub_str(self):
        return u' '.join([] if self.s is None else self.s)

    def __hash__(self):
        return u'sub: {0}; pre: {1}'.format(self.sub_str(), self.pred_str()).__hash__()

    def __eq__(self, other):
        return self.sub_str() == other.sub_str() and self.pred_str() == other.pred_str()

stanford_parser_inst = None


def analysis_ann(ann_file):
    with codecs.open(ann_file, encoding='utf-8') as read_file:
        path = os.path.dirname(ann_file)
        base = os.path.splitext(os.path.basename(ann_file))[0]
        print(path, base)
        ann = json.load(read_file)
        core_stats = {}
        ht_stats = {}
        x_labels = []
        y_values = []
        ht_x_labels = []
        ht_y_values = []

        ontos_y_labels = []
        ontos_x_labels = []
        ontos_ht_x_labels = []
        ontos_ht_y_labels = []
        for a in ann:
            if 'CoreSc' in a:
                x_labels.append(a['CoreSc'])
                # concept = a['CoreSc']
                # concept_stat = {'freq':1} if concept in core_stats else core_stats[concept]
                # concept_stat['freq'] += 1
            else:
                x_labels.append('Sentence')

            xlabel = x_labels[len(x_labels)-1]
            onto_freq = {}
            if 'ncbo' in a:
                y_values.append(len(a['ncbo']))
                for u in a['ncbo']:
                    onto = pann.getEntityType(u['uri'])
                    onto_freq[onto] = 1 if onto not in onto_freq else 1 + onto_freq[onto]
            else:
                y_values.append(0)

            to_x_labels = []
            to_y_labels = []
            for onto in onto_freq:
                to_x_labels.append(xlabel)
                to_y_labels.append(onto)

            ontos_x_labels += to_x_labels
            ontos_y_labels += to_y_labels

            if 'marked' in a:
                ht_x_labels.append(x_labels[len(x_labels)-1])
                ht_y_values.append(y_values[len(y_values)-1])
                ontos_ht_x_labels += to_x_labels
                ontos_ht_y_labels += to_y_labels

        x_values = []
        core_freq = {}
        labels = []
        ht_x_values = []
        for l in x_labels:
            ci = -1
            if l in core_freq:
                ci = core_freq[l]['id']
                core_freq[l]['freq'] += 1
            else:
                ci = len(core_freq) + 1
                core_freq[l] = {'id': ci, 'freq': 1}
                labels.append(l)
            x_values.append(ci + random.uniform(-0.4, 0.4))
        for l in ht_x_labels:
            ht_x_values.append(core_freq[l]['id']+random.uniform(-0.4, 0.4))

        print(json.dumps([(k, core_freq[k]['freq']) for k in core_freq]))

        plt.clf()
        # plt.plot(x_values, y_values, 'r+', ht_x_values, ht_x_values, 'gx', ms=10)
        all_sents = plt.scatter(x_values, y_values, s=80, facecolors='none', edgecolors='r')
        ht_sents = plt.scatter(ht_x_values, ht_x_values, s=80, facecolors='none', edgecolors='g', marker='^')
        plt.xticks(range(1, len(labels)+1), labels)
        plt.axis([0, len(labels) + 1, -1, max(y_values) + 1])
        plt.ylabel('#annotations')
        plt.xlabel('types of sentences')
        plt.legend([all_sents, ht_sents], ['all sentences', 'highlighted'], loc=2)
        # plt.show()
        plt.savefig(os.path.join(path, base + '_all.pdf'))

        onto_x_values = []
        onto_y_values = []
        onto_ht_x_values = []
        onto_ht_y_values = []
        for l in ontos_x_labels:
            onto_x_values.append(core_freq[l]['id'])
        onto_freq = {}
        onto_yaxis_labels = []
        for l in ontos_y_labels:
            ci = -1
            if l in onto_freq:
                ci = onto_freq[l]['id']
                onto_freq[l]['freq'] += 1
            else:
                ci = len(onto_freq) + 1
                onto_freq[l] = {'id': ci, 'freq': 1}
                onto_yaxis_labels.append(l)
            onto_y_values.append(ci + random.uniform(-0.3, 0.3))

        for l in ontos_ht_x_labels:
            onto_ht_x_values.append(core_freq[l]['id'])
        for l in ontos_ht_y_labels:
            onto_ht_y_values.append(onto_freq[l]['id'] + random.uniform(-0.3, 0.3))

        plt.clf()
        all_sents = plt.scatter(onto_x_values, onto_y_values, s=80, facecolors='none', edgecolors='r')
        ht_sents = plt.scatter(onto_ht_x_values, onto_ht_y_values, s=80, facecolors='none', edgecolors='g', marker='^')
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.yticks(range(1, len(onto_yaxis_labels) + 1), onto_yaxis_labels)
        plt.axis([0, len(labels) + 1, 0, len(onto_yaxis_labels) + 1])
        plt.ylabel('ontologies')
        plt.xlabel('types of sentences')
        plt.legend([all_sents, ht_sents], ['all sentences', 'highlighted'], loc=2)
        # print(len(ht_x_values))
        plt.savefig(os.path.join(path, base + '_ontos.pdf'))


def plot_two_sets_data(y1_values, y2_values):
    x1_values = [random.uniform(0, 10) for y in y1_values]
    x2_values = [random.uniform(0, 10) for y in y2_values]
    plt.clf()
    all_sents = plt.scatter(x1_values, y1_values, s=80, facecolors='none', edgecolors='r')
    ht_sents = plt.scatter(x2_values, y2_values, s=80, facecolors='none', edgecolors='g', marker='^')
    plt.show()


def extract_cd_nouns_nes(ht, cd_nouns, name_entities, noun_evidence=None, ne_evd=None):
    text = nltk.word_tokenize(ht.replace('\n', '').strip())
    pr = nltk.pos_tag(text)
    namedEnt = nltk.ne_chunk(pr, binary=True)
    for ent in namedEnt:
        if type(ent) == nltk.tree.Tree and ent.label() == 'NE':
            e = ent[0][0]
            name_entities[e] = 1 if e not in name_entities else 1 + name_entities[e]
            if ne_evd is not None and name_entities[e] == 1:
                ne_evd[e] = ht

    pr_str = ''
    for i in range(len(pr)):
        pr_str += pr[i][1] + str(i) + ' '
    # print(pr_str)
    pr_str = pr_str.strip()
    so_it = re.finditer(r'CD\d+ ((NN\d+|NNS\d+|VBG\d+|JJ\d+|RB\d+|NNP\d+|NNPS\d+|CC\d+) )*(NN|NNS|NNP|NNPS)(\d+)',
                        pr_str, re.M | re.I)
    for so in iter(so_it):
        n = pr[int(so.group(4).strip())][0]
        cd_nouns[n] = 1 if n not in cd_nouns else 1 + cd_nouns[n]
        if noun_evidence is not None and cd_nouns[n] == 1:
            noun_evidence[n] = ht.replace('\n', '').strip()


def analyse_highlighted_text(ht_file):
    anns = None
    with codecs.open(ht_file, encoding='utf-8') as rf:
        anns = json.load(rf)

    n_freqs = {}
    n_evds = {}
    nes = {}
    nes_dvds = {}

    for ann in anns:
        hts = ann['marked']
        for ht in hts:
            extract_cd_nouns_nes(ht, n_freqs, nes, noun_evidence=n_evds, ne_evd=nes_dvds)

    serialise_text_file(n_freqs, n_evds, 'cardinal_noun.txt')
    serialise_text_file(nes, nes_dvds, 'named_entities.txt')

    print('cardinal noun and named entity patterns saved')

    sp_container = {}
    utils.multi_thread_tasking(anns, 15, analysis_sentence_struct, args=[sp_container],
                               callback_func=serialise_pred_obj_json)


def serialise_text_file(data_dict, evd_dict, file_name):
    with codecs.open(os.path.join(pattern_output_folder, file_name), 'w', encoding='utf-8') as wf:
        wf.write('\n'.join([u'{0}\t{1}\t{2}'.format(e, data_dict[e], evd_dict[e]) for e in data_dict]))


def serialise_pred_obj_json(sub_preds):
    sps = sorted([(k.__dict__, sub_preds[k]) for k in sub_preds], cmp=lambda sp1, sp2: sp2[1] - sp1[1])
    with codecs.open(os.path.join(pattern_output_folder, 'sub_pred.json'), 'w', encoding='utf-8') as wf:
        json.dump(sps, wf, encoding='utf-8')
    print('all done')


# create a new stanford parser instance
def create_stanford_parser_inst():
    return StanfordParser(
        model_path=stanford_language_model_file)


# parse the annotation to get its subject-predicate pattern
def analysis_sentence_struct(ann, container=None):
    s = ann['text']
    global stanford_parser_inst
    parser = create_stanford_parser_inst() \
        if stanford_parser_inst is None else \
        stanford_parser_inst
    analysis_sentence_text(parser, s, container)


def analysis_sentence_text(parser, s, container=None):
    sentences = parser.raw_parse(s)
    for line in sentences:
        for sentence in line:
            # print line
            if line.label() == 'ROOT':
                if 'S' == line[0].label():
                    sub = None
                    nps, p = get_pos_from_tree(line[0], r'NP')
                    if nps is not None:
                        noun_nodes, p = get_pos_from_tree(nps[0], r'(NN.*|PRP.*)', get_all=True)
                        if noun_nodes is not None:
                            sub = [n[0] for n in noun_nodes]

                    pred = []
                    p = line[0]
                    vps, p = get_pos_from_tree(p, r'VP', get_all=True)
                    for vp in vps:
                        keep_finding_vps(vp, pred)
                    print(sub, pred)

                    sp = SubjectPredicate(sub, pred)
                    if container is not None:
                        with thread_lock:
                            container[sp] = 1 if sp not in container else 1 + container[sp]
                    return sub, pred
    return None, None


def keep_finding_vps(vp, pred):
    if vp is None:
        return
    for child in vp:
        if re.match(r'VB.*', child.label(), re.M|re.I):
            pred.append(child[0])
        if re.match(r'VP', child.label(), re.M|re.I):
            keep_finding_vps(child, pred)


def get_pos_from_tree(p, pos_pattern, get_all=None):
    if not isinstance(p, nltk.tree.Tree):
        return None, None

    all_matched = []
    for child in p:
        if not isinstance(child, nltk.tree.Tree):
            continue
        if re.match(pos_pattern, child.label(), re.M|re.I):
            all_matched.append(child)
            if get_all is None:
                return all_matched, p
    if len(all_matched) > 0:
        return all_matched, p

    # if it comes to this, we need to keep searching descendants
    for sib in p:
        ret, p = get_pos_from_tree(sib, pos_pattern, get_all)
        if None is not ret:
            return ret, p

    return None, None


def sort_sub_pred(sp_file):
    sps = None
    with codecs.open(sp_file, encoding='utf-8') as rf:
        sps = json.load(rf)
    sps = sorted(sps, cmp=lambda sp1, sp2: sp2[1] - sp1[1])
    print(json.dumps(sps))

if __name__ == "__main__":
    analyse_highlighted_text('./training/hts.json')
    # sort_sub_pred('./training/sub_pred.json')

