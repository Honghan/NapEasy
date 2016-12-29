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
import threading
import pickle
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import auto_highlighter as ah
import math
import ann_utils as utils
from nltk.tag.stanford import StanfordNERTagger


# the lock for gain access to the shared variable
thread_lock = threading.Lock()

# stanford model file path
stanford_language_model_file = "/Users/jackey.wu/Documents/working/libraries/" \
                   "stanford-english-corenlp-2016-01-10-models/" \
                   "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"

# pattern output folder
pattern_output_folder = './training/'

# cardinal english words
cardinal_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four", "twenty-five", "twenty-six", "twenty-seven", "twenty-eight", "twenty-nine", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "one hundred", "a hundred and one", "a hundred and ten", "a hundred and twenty", "two hundred"]


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


def plot_two_sets_data(value_pairs, file_to_save=None):

    plt.clf()
    i = 0
    f, axarr = plt.subplots(len(value_pairs), sharex=True)
    for pair in value_pairs:
        x1_values = [random.uniform(0, 10) for y in pair[0]]
        x2_values = [random.uniform(0, 10) for y in pair[1]]
        axarr[i].scatter(x1_values, pair[0], s=80, facecolors='none', edgecolors='r')
        axarr[i].scatter(x2_values, pair[1], s=80, facecolors='none', edgecolors='g', marker='^')
        if len(pair) >= 3:
            axarr[i].set_title(pair[2])
        i += 1
    if file_to_save is not None:
        pickle.dump(axarr, file(file_to_save, 'w'))
    plt.show()


def replace_cardinal_english_words(text):
    pp = re.compile('|'.join(['({})'.format(re.escape(w)) for w in cardinal_words[::-1]]), re.IGNORECASE)
    return pp.sub('123', text)


def match_ne_dictionary(tokens, nes):
    lst = [t.lower() for t in tokens]
    matched = []
    for k in nes:
        if len(nes[k]) > 1:
            # do case incensitive when multiple words in the named entity
            if contains_sublist(lst, [w.lower() for w in nes[k]]):
                matched.append(k)
        elif len(nes[k]) == 1:
            # otherwise, do case sensitive matching
            if contains_sublist(tokens, nes[k]):
                matched.append(k)
    return matched


def contains_sublist(lst, sublst):
    n = len(sublst)
    return any((sublst == lst[i:i+n]) for i in xrange(len(lst)-n+1))


def get_ht_named_entity_dictionary():
    hter = ah.HighLighter.get_instance()
    ne2list = {}
    for ne in hter.get_named_entities():
        ne2list[ne] = [w for w in ne.split(' ')]
    return ne2list


def extract_cd_nouns_nes(ht, cd_nouns, name_entities, noun_evidence=None, ne_evd=None, ne_dict=None):
    text = nltk.word_tokenize(replace_cardinal_english_words(ht.replace('\n', '').strip()))
    pr = nltk.pos_tag(text)
    namedEnt = nltk.ne_chunk(pr, binary=True)
    for ent in namedEnt:
        if type(ent) == nltk.tree.Tree and ent.label() == 'NE':
            e = u' '.join(e[0] for e in ent)
            name_entities[e] = 1 if e not in name_entities else 1 + name_entities[e]
            if ne_evd is not None and name_entities[e] == 1:
                ne_evd[e] = ht
    if ne_dict is not None:
        m_nes = match_ne_dictionary(text, ne_dict)
        for new_ne in m_nes:
            if new_ne not in name_entities:
                name_entities[new_ne] = 1
    pr_str = ''
    for i in range(len(pr)):
        pr_str += pr[i][1] + str(i) + ' '
    # print(pr_str)
    pr_str = pr_str.strip()
    so_it = re.finditer(r'CD\d+ ((NN\d+|NNS\d+|VBG\d+|VBN\d+|JJ\d+|RB\d+|NNP\d+|NNPS\d+|CC\d+) )*(NN|NNS|NNP|NNPS)(\d+)',
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

    # sp_container = {}
    # utils.multi_thread_tasking(anns, 15, analysis_sentence_struct, args=[sp_container],
    #                            callback_func=serialise_pred_obj_json)


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


# parse the annotation results to extract geometric features
def geometric_analysis(ann_file, container, out_file, highlighter):
    p, fn = os.path.split(ann_file)

    score_file = os.path.join('./summaries/', fn[0:fn.rfind('.')] + '_scores.json')
    scores = utils.load_json_data(score_file)
    sent_scores = {}
    for s in scores:
        sent_scores[s['sid']] = s
    anns = utils.load_json_data(ann_file)
    ht_obj = {'total': len(anns), 'ht_sids': [], 'sect_dict': {}, 'sects': {},
              'page_dict': {}, 'total_page': 0, 'id': ann_file, 'sid_cat':{}}
    sect = ''
    last_sid = ''
    for ann in anns:
        if 'marked' in ann and len(ann['marked'])>0:
            ht_obj['ht_sids'].append(ann['sid'])
            if 'struct' in ann:
                ht_obj['sect_dict'][ann['struct']] = [ann['sid']] if ann['struct'] not in ht_obj['sect_dict'] else \
                    ht_obj['sect_dict'][ann['struct']] + [ann['sid']]
            if 'page' in ann:
                ht_obj['page_dict'][ann['page']] = [ann['sid']] if ann['page'] not in ht_obj['page_dict'] else \
                    ht_obj['page_dict'][ann['page']] + [ann['sid']]
            ht_obj['sid_cat'][ann['sid']] = highlighter.get_sentence_cat_bd(sent_scores[ann['sid']])
        if 'page' in ann:
            ht_obj['total_page'] = ann['page']
        if ann['struct'] != sect:
            if sect.strip() != '':
                ht_obj['sects'][sect]['end'] = last_sid
            sect = ann['struct']
            ht_obj['sects'][ann['struct']] = {'star': ann['sid']}
        last_sid = ann['sid']
        if int(ann['sid']) > ht_obj['total']:
            ht_obj['total'] = int(ann['sid'])

    ht_obj['sects'][sect]['end'] = last_sid
    sum_file = os.path.join('./summaries/', fn[0:fn.rfind('.')] + '.sum')
    sum = utils.load_json_data(sum_file)
    if 'journal' in sum:
        ht_obj['journal'] = sum['journal']
    else:
        ht_obj['journal'] = 'J.'
    container.append(ht_obj)


# used as call back functions when all (multi-threaded processed) geometric features are put in the container
def post_process_geometric_analysis(container, output_file, hter):
    print json.dumps(container)
    utils.save_json_array(container, output_file)
    print 'geometric features of all annotations extracted and saved'


# entry function to extract geometric features from annotations
def extract_geometrics(annotation_files_path, gm_feature_output_file):
    ret_container = []
    hter = ah.HighLighter.get_instance()
    utils.multi_thread_process_files(annotation_files_path, '', 10, geometric_analysis,
                                     args=[ret_container, gm_feature_output_file, hter],
                                     file_filter_func=lambda fn: fn.endswith('_ann.json'),
                                     callback_func=post_process_geometric_analysis)


def visualise_highlights_geometric(geo_feature_file, fn, cat):
    gms = utils.load_json_data(geo_feature_file)
    subplots = {}
    for paper in gms:
        j = paper['journal']
        if j not in subplots:
            subplots[j] = []
        traces = subplots[j]
        y_vals = []
        x_vals = []
        sects = paper['sect_dict']
        sid_cat = paper['sid_cat']
        for y in sects:
            for x in sects[y]:
                if sid_cat[x] == cat:
                    x_vals.append(1.0 * int(x) / int(paper['total']))
                    y_vals.append(y)
        traces.append({'x': x_vals, 'y': y_vals})
    plots = []
    for j in subplots:
        if len(subplots[j]) >= 6 and j is not None:
            m_x = []
            m_y = []
            for d in subplots[j]:
                m_x += d['x']
                m_y += d['y']
            plots.append(go.Scatter(
                x=m_x,
                y=m_y,
                mode='markers',
                name=j if j is not None else 'unknown'
            ))

    fig = tools.make_subplots(rows=len(plots), cols=1, shared_xaxes=True)
    for i in range(len(plots)):
        fig.append_trace(plots[i], i + 1, 1)
    fig['layout'].update(height=600, width=600)
    py.plot(fig, filename=fn)


def visualise_categorised_geometric(geo_feature_file, fn):
    gms = utils.load_json_data(geo_feature_file)
    journal2cat = {}
    journal2papers = {}
    # cat_trace = {}

    for paper in gms:
        # j = paper['journal']
        j = 'all'
        journal2cat[j] = {} if j not in journal2cat else journal2cat[j]
        cat_trace = journal2cat[j]
        journal2papers[j] = [j, 1] if j not in journal2papers else [j, 1 + journal2papers[j][1]]
        sects = paper['sect_dict']
        sid_cat = paper['sid_cat']
        for y in sects:
            for x in sects[y]:
                cat = sid_cat[x]
                if cat in ['cardinal nouns', 'named entities', 'general']:
                    continue
                if cat not in cat_trace:
                    cat_trace[cat] = {'x':[], 'y':[]}
                trace = cat_trace[cat]
                trace['x'].append(1.0 * int(x) / int(paper['total']))
                label_y = y.replace('deo:', '').replace('DoCO:', '').replace('BodyMatter', 'Others').replace('FrontMatter', 'Others')
                trace['y'].append(label_y)

    sorted_journals = sorted([journal2papers[j] for j in journal2papers], cmp=lambda jp1, jp2 : jp2[1] - jp1[1])
    print sorted_journals
    print len(sorted_journals)

    # selected_j = sorted_journals[1][0]
    selected_j = 'all'
    cat_trace = journal2cat[selected_j] # skip the no-journal paper group
    traces = []
    for cat in cat_trace:
        traces.append(go.Scatter(
                x=cat_trace[cat]['x'],
                y=cat_trace[cat]['y'],
                mode='markers',
                name=cat
            ))
    # print traces
    layout = go.Layout(
        title= 'highlights over spatial dimensions', # selected_j + ' - language pattern breakdown',
        yaxis=dict(
            categoryorder = 'array',
            categoryarray = ['Introduction', 'Methods', 'Results', 'Discussion', 'Others']
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    # py.plot(fig, filename=fn) # + ' - ' + selected_j)
    py.image.save_as({'data': traces, 'layout': layout}, './results/spatial.pdf')



def get_general_highlights():
    geos = utils.load_json_data('./training/geo_features.json')
    sents = []
    for g in geos:
        f_ann = g['id']
        sids = []
        for sid in g['sid_cat']:
            if g['sid_cat'][sid] == 'general':
                sids.append(sid)
        if len(sids) > 0:
            anns = utils.load_json_data(f_ann)
            for ann in anns:
                if ann['sid'] in sids:
                    sents.append({'text': ann['text'], 'marked': ann['marked'] if 'marked' in ann else ''})
    utils.save_json_array(sents, './training/general_highlights.json')


def get3DCords(score_file, container, out_file, hter):
    scores = utils.load_json_data(score_file)
    anns = utils.load_json_data(scores[0]['doc_id'])
    sids = []
    for ann in anns:
        if 'marked' in ann:
            sids.append(ann['sid'])
    for s in scores:
        if s['sid'] not in sids:
            continue
        cat = hter.get_sp_type(s)
        p = s['pattern']
        nes = sorted(list(set([k for k in p['nes']])))
        cds = sorted(list(set([k for k in p['cds']])))
        container.append({'x': cat,
            #                   'N/A' if 'sp_index' not in p or p['sp_index'] == -1 else \
            # '-'.join(p['sub'] if p['sub'] is not None else []) + ' ' + \
            # '-'.join(p['pred'] if p['pred'] is not None else []),
                          'y': len(nes),
                          'z': len(cds)
                          # 'y': 'N/A' if len(p['nes']) == 0 else ' '.join(nes),
                          # 'z': 'N/A' if len(p['cds']) == 0 else ' '.join(cds),
                          })


def pp_3D(container, out_file, hter):
    x = []
    y = []
    z = []
    marker2freq = {}
    max_freq = 0
    keys = []
    for p in container:
        k = '{} {} {}'.format(p['x'], p['y'], p['z'])
        if k not in marker2freq:
            x.append(p['x'])
            y.append(p['y'])
            z.append(p['z'])
            marker2freq[k] = 1
            keys.append(k)
        else:
            marker2freq[k] += 1
        if marker2freq[k] > max_freq:
            max_freq = marker2freq[k]

    print 'max freq is %s ' % max_freq
    print json.dumps(marker2freq)
    markers = []
    for k in keys:
        markers.append(int(math.log(1024 * marker2freq.get(k), 2)))
    trace2 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=markers,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        )
    )
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
            xaxis=dict(
                title='Sub-Pred Type'
            ),
            yaxis=dict(
                title='#Named Entities'
            ),
            zaxis=dict(
                title='#Cardinal Nouns'
            )
        )
    )

    fig = go.Figure(data=[trace2], layout=layout)
    py.plot(fig, filename='3D')
    # utils.save_json_array(container, out_file)


def visualise_highlights_3D(annotation_files_path, out_file):
    ret_container = []
    hter = HighLighter.get_instance()
    utils.multi_thread_process_files(annotation_files_path, '', 10, get3DCords,
                                     args=[ret_container, out_file, hter],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_3D)


def get_stats_obj():
    return {'ht': {'sp': {}, 'ne': {}, 'cd': {}, 'sp_breakdown':{}}, 'nm': {'sp': {}, 'ne': {}, 'cd': {}, 'sp_breakdown':{}}, 's_nm': 0, 's_ht': 0}


def get_language_pattern_stats(score_file, container, out_file, hter):
    scores = utils.load_json_data(score_file)
    max_sid = int(scores[len(scores) - 1]['sid'])
    units = 5
    offset = int(1.0 * max_sid / units)
    anns = utils.load_json_data(scores[0]['doc_id'])

    b_marked = False
    ranges = []
    r = {'sids': [], 's': 0, 'seq': 0}
    ranges.append(r)
    for i in range(len(anns)):
        if (i + 1) % offset == 0:
            r['e'] = i - 1
            r = {'sids': [], 's': i, 'seq': (i + 1) / offset}
            ranges.append(r)
        ann = anns[i]
        if 'marked' in ann:
            b_marked = True
            r['sids'].append(ann['sid'])
    r['e'] = len(anns) - 1

    if not b_marked:
        return

    for r in ranges:
        sids = r['sids']
        stats = get_stats_obj()
        stats['s_nm'] = r['e'] - r['s'] - len(sids)
        stats['s_ht'] = len(sids)
        for i in range(r['s'], r['e']):
            s = scores[i]
            sent_type = 'ht' if s['sid'] in sids else 'nm'
            stat = stats[sent_type]['sp']

            all_sp_types = []
            cat = hter.get_sp_type(s, all_types=all_sp_types)
            if len(all_sp_types)>0:
                t = '-'.join(sorted(all_sp_types))
                stat[t] = 1 if t not in stat else 1 + stat[t]
            else:
                # count not typed as well
                stat[cat] = 1 if cat not in stat else 1 + stat[cat]
            p = s['pattern']
            nes = sorted(list(set([k for k in p['nes']])))
            cds = sorted(list(set([k for k in p['cds']])))

            if len(all_sp_types) > 0:
                sp = '-'.join(p['sub'] if p['sub'] is not None else '') + ' ' + '-'.join(p['pred'] if p['pred'] is not None else '')
                stat = stats[sent_type]['sp_breakdown']
                stat[sp] = 1 if sp not in stat else 1 + stat[sp]

            stat = stats[sent_type]['ne']
            for ptn in nes:
                if ptn in hter.get_named_entities():
                    stat[ptn] = 1 if ptn not in stat else 1 + stat[ptn]
            stat = stats[sent_type]['cd']
            for ptn in cds:
                if ptn in hter.get_cardinal_nouns():
                    stat[ptn] = 1 if ptn not in stat else 1 + stat[ptn]
        container.append({'r%s' % r['seq']: stats})


def merge_key_freq(container, data, l1, l2):
    for ptn in data[l1][l2]:
        freq = data[l1][l2][ptn]
        m = container[l1][l2]
        m[ptn] = freq if ptn not in m else m[ptn] + freq


def pp_pattern_stats(container, out_file, hter):
    range2stats = {}
    for stats in container:
        for k in stats:
            range2stats[k] = [stats[k]] if k not in range2stats else [stats[k]] + range2stats[k]

    range2merged = {}
    for r in range2stats:
        merged = {'ht': {'sp': {}, 'ne': {}, 'cd': {}, 'sp_breakdown':{}}, 'nm': {'sp': {}, 'ne': {}, 'cd': {}, 'sp_breakdown':{}}, 's_ht': 0, 's_nm': 0}
        for stats in range2stats[r]:
            merge_key_freq(merged, stats, 'ht', 'sp')
            merge_key_freq(merged, stats, 'ht', 'ne')
            merge_key_freq(merged, stats, 'ht', 'cd')
            merge_key_freq(merged, stats, 'ht', 'sp_breakdown')
            merge_key_freq(merged, stats, 'nm', 'sp')
            merge_key_freq(merged, stats, 'nm', 'ne')
            merge_key_freq(merged, stats, 'nm', 'cd')
            merge_key_freq(merged, stats, 'nm', 'sp_breakdown')
            merged['s_ht'] += stats['s_ht']
            merged['s_nm'] += stats['s_nm']
        range2merged[r] = merged
    utils.save_json_array(range2merged, out_file)


def analyse_language_pattern_stats(score_files_path, out_file):
    ret_container = []
    hter = ah.HighLighter.get_instance()
    utils.multi_thread_process_files(score_files_path, '', 10, get_language_pattern_stats,
                                     args=[ret_container, out_file, hter],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_pattern_stats)


def score_language_patterns(normals, highlights, num_normals, num_highlights):
    epsiton = 0.015
    keys = sorted([k for k in highlights])
    scores = []
    for key in keys:
        ht = 1.0 * highlights[key]/num_highlights
        nm = 0.0 if key not in normals else 1.0 * normals[key]/num_normals
        scores.append(math.log((ht + epsiton) / (nm + epsiton), 2))
    return keys, scores


def visualise_lp_stats(stat_file, cat, title, skips=None, score_output_file=None):
    stats = utils.load_json_data(stat_file)
    total_normal = stats['s_nm']
    total_highlights = stats['s_ht']
    keys, scores = score_language_patterns(stats['nm'][cat], stats['ht'][cat],
                                           total_normal, total_highlights)
    if score_output_file is None:
        trace1 = go.Bar(
            x=keys,
            y=scores,
            name='Highlighted Sentences / Other Sentences'
        )
        data = [trace1]
        layout = go.Layout(
            barmode='group',
            title=title
        )

        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='language pattern stats - ' + cat)
    else:
        data = {}
        for i in range(len(keys)):
            data[keys[i]] = scores[i]
        utils.save_json_array(data, score_output_file)


def visualise_lp_ranged_stats(stat_file, cat, title, skips=None, score_output_file=None):
    r2stats = utils.load_json_data(stat_file)
    data = []
    data2save = {}
    for r in r2stats:
        stats = r2stats[r]
        total_normal = stats['s_nm']
        total_highlights = stats['s_ht']
        keys, scores = score_language_patterns(stats['nm'][cat], stats['ht'][cat],
                                           total_normal, total_highlights)
        if score_output_file is None:
            trace1 = go.Bar(
                x=keys,
                y=scores,
                name=r
            )
            data.append(trace1)
        else:
            data2save[r] = {}
            for i in range(len(keys)):
                data2save[r][keys[i]] = scores[i]
    if score_output_file is None:
        layout = go.Layout(
            barmode='group',
            title=title
        )

        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename='language pattern ranged stats - ' + cat)
    else:
        utils.save_json_array(data2save, score_output_file)


def plot_ly_login():
    tools.set_credentials_file(username='honghan.wu', api_key='gy90jemd3t')


# doing paper-wise language pattern distribution and highlighted sentence distribution analysis
# the idea is to categorise papers to guide the highlights generation - e.g., how many findings
# need to be generated for a particular sentence
def paper_language_pattern_dist(score_file, container, hter, out_file):
    scores = utils.load_json_data(score_file)
    anns = utils.load_json_data(scores[0]['doc_id'])

    b_marked = False
    hts = []
    for i in range(len(anns)):
        ann = anns[i]
        if 'marked' in ann:
            b_marked = True
            hts.append(ann['sid'])

    if not b_marked or 15 > len(hts) < 10:
        return

    max_sid = int(scores[len(scores) - 1]['sid'])
    stat = {'ht': {}, 'all': {}, 'max_sid': max_sid}
    for s in scores:
        all_sp_types = []
        cat = hter.get_sp_type(s, all_types=all_sp_types)
        for t in all_sp_types:
            stat['all'][t] = 1 if t not in stat['all'] else 1 + stat['all'][t]
            if s['sid'] in hts:
                stat['ht'][t] = 1 if t not in stat['ht'] else 1 + stat['ht'][t]
        p = s['pattern']
        if len(p['nes']) > 0:
            t = 'NE'
            stat['all'][t] = 1 if t not in stat['all'] else 1 + stat['all'][t]
            if s['sid'] in hts:
                stat['ht'][t] = 1 if t not in stat['ht'] else 1 + stat['ht'][t]
        if len(p['cds']) > 0:
            t = 'CDS'
            stat['all'][t] = 1 if t not in stat['all'] else 1 + stat['all'][t]
            if s['sid'] in hts:
                stat['ht'][t] = 1 if t not in stat['ht'] else 1 + stat['ht'][t]
    container.append(stat)


def lp_dist_cb(ctn, hter, out_file):
    print json.dumps(ctn)
    x = []
    y = []
    pt2freq = {}
    keys = []

    goals = 0
    methods = 0
    findings = 0
    all = 0
    for p in ctn:
        g = 0 if 'goal' not in p['ht'] else p['ht']['goal']
        m = 0 if 'method' not in p['ht'] else p['ht']['method']
        f = 0 if 'findings' not in p['ht'] else p['ht']['findings']
        all += p['max_sid']
        goals += g
        methods += m
        findings += f
        print '{}\t{}\t{}'.format(
            g,
            m,
            f)
        x1 = 0 if 'method' not in p['ht'] else p['ht']['method'] #round(p['ht']['method'] * 1.0 / p['max_sid'], 4))
        y1 = 0 if 'method' not in p['all'] else p['all']['method'] #round(p['ht']['findings'] * 1.0 / p['max_sid'], 4))
        x.append(x1)
        y.append(y1)
        k = '{} {}'.format(x1, y1)
        pt2freq[k] = 1 if k not in pt2freq else 1 + pt2freq[k]
        keys.append(k)
    print '{}\t{}\t{}\t{}'.format(1.0*goals/len(ctn), 1.0*methods/len(ctn), 1.0*findings/len(ctn), 1.0*all/len(ctn))
    markers = []
    for k in keys:
        markers.append(pt2freq[k] + 3)

    trace = go.Scatter(
        x=x,
        y=y,
        marker=dict(
            size=markers,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        ),
        mode='markers',
        name='LP Dist'
    )
    data = [trace]
    py.plot(data, filename='LP Dist')


def lp_dist_cal(score_files_path, out_file):
    ret_container = []
    hter = ah.HighLighter.get_instance()
    utils.multi_thread_process_files(score_files_path, '', 10, paper_language_pattern_dist,
                                     args=[ret_container, hter, out_file],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=lp_dist_cb)


def compute_sp_type_statics():
    sp2ratio = {}
    stats = utils.load_json_data('./training/language_pattern_stats_ranged.json')
    total = 0
    for r in stats:
        total += stats[r]['s_ht'] + stats[r]['s_nm']
        for p in stats[r]['ht']['sp']:
            print p, stats[r]['ht']['sp'][p]
            sp2ratio[p] = stats[r]['ht']['sp'][p] if p not in sp2ratio else stats[r]['ht']['sp'][p] + sp2ratio[p]

    print json.dumps(sp2ratio)
    for p in sp2ratio:
        sp2ratio[p] = sp2ratio[p] * 1.0 / total
    print json.dumps(sp2ratio)


def compute_sp_type_regioned_weights():
    sp2ratio = {}
    stats = utils.load_json_data('./training/language_pattern_stats_ranged.json')
    total = 0
    for r in stats:
        total += stats[r]['s_ht'] + stats[r]['s_nm']
        for p in stats[r]['ht']['sp']:
            sp2ratio[p] = {} if p not in sp2ratio else sp2ratio[p]
            sp2ratio[p][r] = stats[r]['ht']['sp'][p]
            sp2ratio[p]['max'] = stats[r]['ht']['sp'][p] \
                if 'max' not in sp2ratio[p] or sp2ratio[p]['max'] < stats[r]['ht']['sp'][p] \
                else sp2ratio[p]['max']

    for p in sp2ratio:
        for k in sp2ratio[p]:
            m = sp2ratio[p]['max']
            if k != 'max':
                sp2ratio[p][k] = 1.0 * sp2ratio[p][k] / m

    print json.dumps(sp2ratio)


# compute the ncbo stats in highlighted and non-highlighted sentences
def get_ncbo_stats(ann_file, container):
    anns = utils.load_json_data(ann_file)
    onto2freq = {'ht': {}, 'nm': {}}
    total_nm = 0
    total_ht = 0
    for ann in anns:
        if 'marked' in ann:
            total_ht += 1
        else:
            total_nm += 1
        if 'ncbo' in ann:
            matched_ontos = []
            for ncbo in ann['ncbo']:
                for name in pann.onto_name:
                    if name not in matched_ontos and ncbo['uri'].startswith(pann.onto_name[name]):
                        matched_ontos.append(name)
                    if name in matched_ontos:
                        break
            # for name in matched_ontos:
            #     ctn = onto2freq['ht'] if 'marked' in ann else onto2freq['nm']
            #     ctn[name] = 1 if name not in ctn else 1 + ctn[name]
            if len(matched_ontos) > 0:
                comb = '-'.join(sorted(matched_ontos))
                ctn = onto2freq['ht'] if 'marked' in ann else onto2freq['nm']
                ctn[comb] = 1 if comb not in ctn else 1 + ctn[comb]
    container.append({'total_nm': total_nm, 'total_ht': total_ht, 'freqs': onto2freq})


def pp_ncbo_stat(container):
    t_nm = 0
    t_ht = 0
    overall_freqs = {'ht':{}, 'nm':{}}
    for c in container:
        t_ht += c['total_ht']
        t_nm += c['total_nm']
        for k in c['freqs']['ht']:
            p = c['freqs']['ht']
            overall_freqs['ht'][k] = p[k] if k not in overall_freqs['ht'] else p[k] + overall_freqs['ht'][k]
        for k in c['freqs']['nm']:
            p = c['freqs']['nm']
            overall_freqs['nm'][k] = p[k] if k not in overall_freqs['nm'] else p[k] + overall_freqs['nm'][k]

    scores = {}
    for k in overall_freqs['nm']:
        overall_freqs['nm'][k] = 1.0 * overall_freqs['nm'][k] / t_nm
    for k in overall_freqs['ht']:
        overall_freqs['ht'][k] = 1.0 * overall_freqs['ht'][k] / t_ht
        scores[k] = math.log(overall_freqs['ht'][k] / overall_freqs['nm'][k], 2)
    print json.dumps(scores)


    # trace1 = go.Bar(
    #     x=[k for k in overall_freqs['ht']],
    #     y=[overall_freqs['ht'][k] for k in overall_freqs['ht']],
    #     name='Highlighted'
    # )
    # trace2 = go.Bar(
    #     x=[k for k in overall_freqs['nm']],
    #     y=[overall_freqs['nm'][k] for k in overall_freqs['nm']],
    #     name='Others'
    # )
    # data = [trace1, trace2]
    # py.plot(data, filename='NCBO Onto Distribute-180')


def compute_overall_ncbo_stat(ann_files_path):
    ret_container = []
    utils.multi_thread_process_files(ann_files_path, '', 10, get_ncbo_stats,
                                     args=[ret_container],
                                     file_filter_func=lambda fn: fn.endswith('_ann.json'),
                                     callback_func=pp_ncbo_stat)


def get_sp_ne_associations(score_file, container):
    scores = utils.load_json_data(score_file)
    sp2ne = {}
    for s in scores:
        p = s['pattern']
        if 'sp_index' in p and p['sp_index'] > -1 and s['ne'] > 0:
            sp2ne[p['sp_index']] = 1 if p['sp_index'] not in sp2ne else 1 + sp2ne[p['sp_index']]
    container.append(sp2ne)


def pp_sp_ne_asso(container):
    merged = {}
    for c in container:
        for k in c:
            merged[k] = c[k] if k not in merged else c[k] + merged[k]

    print merged[0]
    print json.dumps(merged)


def compute_sp_ne_stat(score_files_path):
    ret_container = []
    utils.multi_thread_process_files(score_files_path, '', 10, get_sp_ne_associations,
                                     args=[ret_container],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_sp_ne_asso)


# basic stats of the corpus
def paper_stat(ann_file, container):
    path, fn = utils.split(ann_file)
    sums = utils.load_json_data(utils.join('./20-test-papers/summaries/', fn[:fn.rfind('.')] + '.sum'))
    anns = utils.load_json_data(ann_file)
    total_ht = 0
    for ann in anns:
        if 'marked' in ann:
            total_ht += 1
    container.append({'f': ann_file, 'ht': total_ht, 'nm': len(anns) - total_ht, 'total': len(anns),
                      'PMID': sums['PMID'] if 'PMID' in sums else '',
                      'Journal': sums['journal'] if 'journal' in sums else ''})


def pp_paper_stat(ctn):
    print '\n'.join(['{}\t{}\t{}\t{}\t{}\t{}'.format(s['PMID'], s['Journal'], s['total'], s['nm'], s['ht'], s['f']) for s in ctn])
    print json.dumps(ctn)


def corpus_simple_stat(ann_files_path):
    ret_container = []
    utils.multi_thread_process_files(ann_files_path, '', 10, paper_stat,
                                     args=[ret_container],
                                     file_filter_func=lambda fn: fn.endswith('_ann.json'),
                                     callback_func=pp_paper_stat
    )


def remove_ann_sentences(ann_file):
    anns = utils.load_json_data(ann_file)
    for ann in anns:
        ann['text'] = ''
    utils.save_json_array(anns, ann_file)


def clean_ann_files(ann_files_path):
    utils.multi_thread_process_files(ann_files_path, '', 10, remove_ann_sentences,
                                     file_filter_func=lambda fn: fn.endswith('_ann.json')
                                     )


if __name__ == "__main__":
    # corpus_simple_stat('./20-test-papers/')
    plot_ly_login()
    visualise_categorised_geometric('./training/geo_features.json', 'ht_geometric_features_categorised')
