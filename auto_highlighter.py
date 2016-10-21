import json
import ann_analysor as aa
import codecs
import ann_utils as utils
from os.path import split, join, isfile
import threading
import numpy as np
import math
import article_ann as pann

res_file_cd = './resources/cardinal_Noun_patterns.txt'  #'./training/patterns/cardinal_noun.txt'
res_file_ne = './resources/named_entities.txt' # './training/patterns/named_entities.txt'
res_file_sp = './resources/sub_pred.txt' # './training/patterns/sub_pred.json'
res_file_spcat = './resources/sub_pred_categories.json'

score_file_cd = './training/scores_cd.json'
score_file_nes = './training/scores_nes.json'
score_file_sp = './training/scores_sp.json'
score_file_ranged_cd = './training/scores_ranged_cd.json'
score_file_ranged_nes = './training/scores_ranged_nes.json'
score_file_ranged_sp = './training/scores_ranged_sp.json'

# parser_lock = threading.RLock()


class HighLighter:
    sent_type_boost = None
    onto_name_scores = None
    sub_pred_ne_stat = None
    brain_regions = None
    sub_pred = None

    def __init__(self, parser, ne_res, cardinal_noun_res, sub_pred_res,
                 score_nes, score_cd, score_sp,
                 score_ranged_nes, score_ranged_cd, score_ranged_sp,
                 sub_pred_cats=None):
        self.ne = ne_res
        self.card = cardinal_noun_res
        self.sp = sub_pred_res
        self.sp_cats = sub_pred_cats
        self.stanford_parser = parser
        self.score_cd = score_cd
        self.score_nes = score_nes
        self.score_sp = score_sp
        self.score_ranged_cd = score_ranged_cd
        self.score_ranged_nes = score_ranged_nes
        self.score_ranged_sp = score_ranged_sp
        self.normalise_all()
        # print('loading stanford parser...')
        # self.stanford_parser = aa.create_stanford_parser_inst()
        # print('stanford parser loaded')

    def normalise_all(self):
        HighLighter.normalise_scores(self.score_cd)
        HighLighter.normalise_scores(self.score_nes)
        HighLighter.normalise_scores(self.score_sp)
        HighLighter.normalise_regioned_scores(self.score_ranged_cd)
        HighLighter.normalise_regioned_scores(self.score_ranged_nes)
        HighLighter.normalise_regioned_scores(self.score_ranged_sp)

    def get_named_entities(self):
        return self.ne

    def get_cardinal_nouns(self):
        return self.card

    def get_score_cd(self):
        return self.score_cd

    def get_score_nes(self):
        return self.score_nes

    def get_score_sp(self):
        return self.score_sp

    def compute_language_patterns(self, sent_text, doc_id=None, sid=None, container=None):
        scores = {'cd': 0, 'ne': 0, 'sp': 0}
        cd_nouns = {}
        named_entities = {}
        aa.extract_cd_nouns_nes(sent_text, cd_nouns, named_entities)
        for cdn in cd_nouns:
            scores['cd'] += 0 if cdn not in self.card else self.card[cdn]
        for ne in named_entities:
            scores['ne'] += 0 if ne not in self.ne else self.ne[ne]
        sub = None
        pred = None
        try:
            slen = len(sent_text)
            if 10 < slen < 500:
                sub = None
                pred = None
                # with parser_lock:
                sub, pred = aa.analysis_sentence_text(self.stanford_parser, sent_text)
                sp = aa.SubjectPredicate(sub, pred)
                scores['sp'] = 0 if sp not in self.sp else self.sp[sp]['freq']
            else:
                scores = {'cd': 0, 'ne': 0, 'sp': 0}
        except:
            print(u'failed parsing sentences for {0}'.format(sent_text))
        if 'sp' not in scores:
            scores['sp'] = 0
        scores['total'] = scores['cd'] + scores['ne'] + scores['sp']
        scores['pattern'] = {'cds': cd_nouns, 'nes': named_entities}
        if sub is not None or pred is not None:
            scores['pattern']['sub'] = sub
            scores['pattern']['pred'] = pred
            scores['pattern']['sp_index'] = -1 if sp not in self.sp else self.sp[sp]['index']

        if doc_id is not None:
            scores['doc_id'] = doc_id
        if sid is not None:
            scores['sid'] = sid
        if container is not None:
            container.append(scores)
        return scores

    def score(self, score_obj, region=None):
        single_score_threshold = 0
        language_patterns = score_obj['pattern']
        sp_score = {}
        nes_score = 0
        cds_score = 0
        all_sp_types = []
        sp_type = self.get_sp_type(score_obj, all_sp_types)

        res_score_sp = self.score_sp
        res_score_cds = self.score_cd
        res_score_nes = self.score_nes
        if region is not None:
            res_score_sp = self.score_ranged_sp[region]
            res_score_cds = self.score_ranged_cd[region]
            res_score_nes = self.score_ranged_nes[region]

        if sp_type != 'No-SP-Cat':
            for t in all_sp_types:
                s = res_score_sp[t] if t in res_score_sp else 0
                if s >= single_score_threshold:
                    sp_score[t] = s
        for ne in language_patterns['nes']:
            s = 0 if ne not in res_score_nes else res_score_nes[ne]
            if s >= single_score_threshold:
                nes_score += s
        for cd in language_patterns['cds']:
            s = 0 if cd not in res_score_cds else res_score_cds[cd]
            if s >= single_score_threshold:
                cds_score += s
        return {'sp': sp_score, 'nes': nes_score, 'cds': cds_score, 'all_sps': all_sp_types}

    def get_vector(self, score_obj, region=None):
        vect = {}
        language_patterns = score_obj['pattern']
        all_sp_types = []
        sp_type = self.get_sp_type(score_obj, all_sp_types)

        res_score_sp = self.score_sp
        res_score_cds = self.score_cd
        res_score_nes = self.score_nes
        if region is not None:
            res_score_sp = self.score_ranged_sp[region]
            res_score_cds = self.score_ranged_cd[region]
            res_score_nes = self.score_ranged_nes[region]

        sp_score = 0
        nes_score = 0
        cds_score = 0

        if sp_type != 'No-SP-Cat':
            for t in all_sp_types:
                s = res_score_sp[t] if t in res_score_sp else 0
                vect[t] = s
                sp_score += s
        for ne in language_patterns['nes']:
            s = 0 if ne not in res_score_nes else res_score_nes[ne]
            vect[ne] = s
            nes_score += s
        for cd in language_patterns['cds']:
            s = 0 if cd not in res_score_cds else res_score_cds[cd]
            vect[cd] = s
            cds_score += s
        return vect, sp_score + nes_score + cds_score

    def get_sentence_cat(self, scores):
        cats = []
        if 'sp_index' in scores['pattern']:
            if scores['pattern']['sp_index'] != -1:
                for cat in self.sp_cats:
                    if scores['pattern']['sp_index'] in self.sp_cats[cat]:
                        cats.append(cat)
        if len(cats) == 0:
            if scores['cd'] > 0 or scores['ne'] > 0:
                cats.append('method')
        if len(cats) == 0:
            cats.append('general')
        return cats[0]

    # get the breakdown category info
    def get_sentence_cat_bd(self, scores):
        cats = []
        if 'sp_index' in scores['pattern']:
            if scores['pattern']['sp_index'] != -1:
                for cat in self.sp_cats:
                    if scores['pattern']['sp_index'] in self.sp_cats[cat]:
                        cats.append(cat)
        if len(cats) == 0:
            if scores['cd'] > 0:
                cats.append('cardinal nouns')
            elif scores['ne'] > 0:
                cats.append('named entities')
        if len(cats) == 0:
            cats.append('general')
        return cats[0]

    # get the breakdown category info
    def get_sp_type(self, scores, all_types=None):
        cats = []
        if 'sp_index' in scores['pattern']:
            if scores['pattern']['sp_index'] != -1:
                for cat in self.sp_cats:
                    if scores['pattern']['sp_index'] in self.sp_cats[cat]:
                        cats.append(cat)
        if len(cats) > 0:
            if all_types is not None:
                all_types += cats[:]
            return cats[0]
        else:
            return 'No-SP-Cat'

    def summarise(self, sentences, src=None, sids=None, score_dict=None):
        threshold = 1
        summary = {}
        i = 0
        scores_list = []
        for sent in sentences:
            cats = []
            sid = None if sids is None else sids[i]
            sent = sent.replace('\n', '').strip()

            scores = score_dict[str(i+1)] if score_dict is not None and sid is not None else \
                self.compute_language_patterns(sent, doc_id=src, sid=sid)
            # scores = self.score(sent, doc_id=src, sid=sid)
            scores['sid'] = str(i+1)
            i += 1
            scores_list.append(scores)
            if scores['total'] < threshold:
                continue
            cat = self.get_sentence_cat(scores)
            summary[cat] = [(sent, scores)] if cat not in summary else summary[cat] + [(sent, scores)]
            # i += 1

        if 'goal' in summary and 'method' in summary and 'general' in summary:
            summary.pop('general', None)

        num_sents_per_cat = 2
        for cat in summary:
            summary[cat] = HighLighter.pick_top_k(summary[cat], 100) if cat == 'findings' \
                else HighLighter.pick_top_k(summary[cat], num_sents_per_cat)
        print json.dumps(summary)
        return summary, scores_list

    @staticmethod
    def normalise_scores(scores):
        m_score = -1000
        for k in scores:
            if scores[k] > m_score:
                m_score = scores[k]
        for k in scores:
            scores[k] = 1.0 * scores[k] / m_score

    @staticmethod
    def normalise_regioned_scores(regioned_scores):
        m_score = -1000
        for r in regioned_scores:
            scores = regioned_scores[r]
            for k in scores:
                if scores[k] > m_score:
                    m_score = scores[k]
        for r in regioned_scores:
            scores = regioned_scores[r]
            for k in scores:
                scores[k] = 1.0 * scores[k] / m_score

    @staticmethod
    def pick_top_k(sents, k):
        if len(sents) == 0:
            return sents

        sorted_sents = sorted(sents, cmp=lambda s1, s2: s2[1]['total'] - s1[1]['total'])
        return sorted_sents[:k] if 'sid' not in sorted_sents[0][1] \
            else sorted(sorted_sents[:k], cmp=lambda s1, s2: int(s1[1]['sid']) - int(s2[1]['sid']))

    # load sentence type boost setting
    @staticmethod
    def get_sent_type_boost():
        if HighLighter.sent_type_boost is None:
            HighLighter.sent_type_boost = utils.load_json_data('./resources/sent_type_region_boost.json')
        return HighLighter.sent_type_boost

    # get onto name scores
    @staticmethod
    def get_onto_name_scores():
        if HighLighter.onto_name_scores is None:
            HighLighter.onto_name_scores = utils.load_json_data('./training/score_ncbo_ontos.json')
        return HighLighter.onto_name_scores

    # get sub_pred_ne_stat
    @staticmethod
    def get_sub_pred_ne_stat():
        if HighLighter.sub_pred_ne_stat is None:
            HighLighter.sub_pred_ne_stat = utils.load_json_data('./resources/sub_pred_ne_stat.json')
        return HighLighter.sub_pred_ne_stat

    @staticmethod
    def get_brain_regions():
        if HighLighter.brain_regions is None:
            HighLighter.brain_regions = utils.load_text_file('./resources/brain-regions.txt')
        return HighLighter.brain_regions

    @staticmethod
    def get_sub_pred():
        if HighLighter.sub_pred is None:
            HighLighter.sub_pred = utils.load_json_data('./resources/sub_pred.txt')
        return HighLighter.sub_pred

    # get the instance of this class
    @staticmethod
    def get_instance():
        parser = aa.create_stanford_parser_inst()
        ne, cd, sp, cats, \
        scores_nes, scores_cds, scores_sp,\
        scores_ranged_nes, scores_ranged_cds, scores_ranged_sp \
            = load_resources(
            res_file_ne, res_file_cd, res_file_sp,
            score_file_nes, score_file_cd, score_file_sp,
            score_file_ranged_nes, score_file_ranged_cd, score_file_ranged_sp,
            res_file_spcat)
        return HighLighter(parser, ne, cd, sp,
                           scores_nes, scores_cds, scores_sp,
                           scores_ranged_nes, scores_ranged_cds, scores_ranged_sp,
                           cats)


def read_text_res(res_file):
    res = {}
    with codecs.open(res_file, encoding='utf-8') as rf:
        for line in rf.readlines():
            arr = line.split('\t')
            res[arr[0]] = int(arr[1])
    return res


def read_sub_pred_file(res_file):
    sp_raw = None
    with codecs.open(res_file, encoding='utf-8') as rf:
        sp_raw = json.load(rf, encoding='utf-8')
    sps = {}
    for i in range(len(sp_raw)):
        sps[aa.SubjectPredicate(sp_raw[i][0]['s'], sp_raw[i][0]['p'])] = {'freq': sp_raw[i][1], 'index': i}
    return sps


def load_resources(ne_file, cd_file, sp_file,
                   sf_nes, sf_cds, sf_sp,
                   sf_ranged_nes, sf_ranged_cds, sf_ranged_sp,
                   sp_cat_file=None):
    ne = read_text_res(ne_file)
    cd = read_text_res(cd_file)
    sp = read_sub_pred_file(sp_file)
    sp_cats = None if sp_cat_file is None else utils.load_json_data(sp_cat_file)
    scores_nes = utils.load_json_data(sf_nes)
    scores_cds = utils.load_json_data(sf_cds)
    scores_sp = utils.load_json_data(sf_sp)
    scores_ranged_nes = utils.load_json_data(sf_ranged_nes)
    scores_ranged_cds = utils.load_json_data(sf_ranged_cds)
    scores_ranged_sp = utils.load_json_data(sf_ranged_sp)
    return ne, cd, sp, sp_cats, \
           scores_nes, scores_cds, scores_sp, \
           scores_ranged_nes, scores_ranged_cds, scores_ranged_sp


def score_sentence(her, item, container, out_file=None):
    her.compute_language_patterns(item['text'], doc_id=item['src'], sid=item['sid'], container=container)


def do_highlight(test_file):
    thread_nums = 5
    hters = []
    for i in range(thread_nums):
        print('initialising highlighter instance...')
        hters.append(HighLighter.get_instance())
        print('highlighter instance initialised')
    data = None
    with codecs.open(test_file, encoding='utf-8') as rf:
        data = json.load(rf)
    scores = []
    out_file = test_file[:test_file.rfind('.')] + "_scores.json"
    print('multithreading...')
    utils.multi_thread_tasking(data, thread_nums, score_sentence, args=[scores, out_file],
                               thread_wise_objs=hters,
                               callback_func=lambda hl, s, of: utils.save_json_array(s, of))
    print('multithreading started')


def do_compare():
    do_highlight('./training/test/non_hts.json')
    do_highlight('./training/test/hts.json')


def visualise_result(f1, f2):
    score1 = utils.load_json_data(f1)
    score2 = utils.load_json_data(f2)
    # aa.plot_two_sets_data([
    #     ([s['cd'] for s in score1], [s['total'] for s in score2], 'Cardinal Nouns'),
    #     ([s['ne'] for s in score1], [s['ne'] for s in score2], 'Named Entities'),
    #     ([s['sp'] for s in score1], [s['ne'] for s in score2], 'Subject/Predicate Patterns'),
    #     ([s['total'] for s in score1], [s['total'] for s in score2], 'sum of all scores'),
    #     ([s['cd'] * 0.4 + s['ne'] * 0.2 + s['sp'] * 0.4 for s in score1], [s['total'] for s in score2], 'weighted scores')
    #     ], './training/test/total_score.pickle')

    for threshold in range(1, 30):
        abv1 = 0
        abv2 = 0
        for s in score1:
            if s['total'] > threshold:
                abv1 += 1
        for s in score2:
            if s['total'] > threshold:
                abv2 += 1
        print "threshold {2} - nht:{0} ht:{1}".format(abv1 * 1.0 / len(score1),
                                                      abv2 * 1.0 / len(score2),
                                                      threshold)


def summ(highlighter, ann_file, out_path):
    anns = utils.load_json_data(ann_file)
    p, fn = split(ann_file)
    score_file = join(out_path, fn[:fn.rfind('.')] + '_scores.json')
    sid_to_score = None
    # if isfile(score_file):
    #     stored_scores = utils.load_json_data(score_file)
    #     i = 1
    #     for score in stored_scores:
    #         sid_to_score[str(i)] = score
    #         i += 1

    summary, scores = highlighter.summarise([s['text'] for s in anns], src=ann_file, sids=[s['sid'] for s in anns],
                                            score_dict=sid_to_score)
    # if not isfile(score_file):
    utils.save_json_array(scores, score_file)
    utils.save_json_array(summary, join(out_path, fn[:fn.rfind('.')] + '.sum'))


def summarise_all_papers(ann_path, summ_path):
    thread_num = 6
    hters = []
    for i in range(thread_num):
        hters.append(HighLighter.get_instance())
    utils.multi_thread_process_files(ann_path, '', thread_num, summ,
                                     args=[summ_path],
                                     thread_wise_objs=hters,
                                     file_filter_func=lambda f: f.endswith('_ann.json'))


def sort_complement(list1, list2, threshold, cmp=None):
    l = sorted(list1, cmp=cmp)
    if len(l) >= threshold:
        return l[:threshold]
    elif len(list2) > 0:
        num_more = threshold - len(l)
        l2 = sorted(list2, cmp=cmp)
        return l + l2[:num_more]
    else:
        return l


def sort_by_threshold(list1, threshold, cmp=None):
    l = sorted(list1, cmp=cmp)
    for i in range(len(l)):
        if l[i][1] < threshold:
            return l[:i]
    return l


def score_paper(score_file, container, out_file, hter, threshold):
    units = 5
    scores = utils.load_json_data(score_file)
    max_sid = int(scores[len(scores) - 1]['sid'])
    offset = int(1.0 * max_sid / units)

    ht_settings = {'goal': threshold['goal'],
                   'findings': int(threshold['findings'] * len(scores)),
                   'method': int(threshold['method'] * len(scores))}

    anns = utils.load_json_data(scores[0]['doc_id'])
    hts = []
    for ann in anns:
        if 'marked' in ann:
            hts.append(ann['sid'])

    if len(hts) == 0:
        return

    prediction = []
    single_typed = {'goal': [], 'method': [], 'findings': []}
    multi_typed = {'goal': [], 'method': [], 'findings': []}
    others = []
    num_correct = 0
    r = 0
    for i in range(len(scores)):
        score = scores[i]
        r = (i + 1) / offset
        score_ret = hter.score(score, region='r' + str(r))
        if len(score_ret['sp']) > 0 or (score_ret['cds'] + score_ret['nes'] > 0):
            s = 0
            other_score = score_ret['cds'] + score_ret['nes']
            voted_t = None
            if len(score_ret['sp']) > 0:
                if len(score_ret['sp']) == 1:
                    for t in score_ret['sp']:
                        single_typed[t].append([score['sid'], score_ret['sp'][t] + other_score])
                        s = score_ret['sp'][t]
                else:
                    type_score = []
                    for t in score_ret['sp']:
                        type_score.append([t, score_ret['sp'][t]])
                    type_score = sorted(type_score, cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1 )
                    multi_typed[type_score[0][0]].append([score['sid'], type_score[0][1] + other_score])
                    s = type_score[0][1]
                    voted_t = type_score[0][0]
            else:
                others.append([score['sid'], score_ret['cds'] + score_ret['nes']])
                s = 0
            print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(score['sid'] in hts, '-'.join(score_ret['sp']), voted_t, s,
                                                    s + other_score, 'r' + str(r),
                                                    score_ret['cds'], score_ret['nes'])

    goals = sort_complement(single_typed['goal'], multi_typed['goal'], ht_settings['goal'],
                            cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1)
    # method = sort_complement(single_typed['method'], multi_typed['method'], ht_settings['method'],
    #                         cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1)
    # findings = sort_complement(single_typed['findings'], multi_typed['findings'], ht_settings['findings'],
    #                         cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1)
    combined = sort_complement([],
                               single_typed['findings'] + multi_typed['findings'] +
                               single_typed['method'] + multi_typed['method'] +
                               others,
                               # max(1, len(hts) - len(goals)),
                               ht_settings['findings'] + ht_settings['method'],
                            cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1)
    for l in [goals, combined]:
        prediction += l
        c = 0
        for s in l:
            if s[0] in hts:
                c += 1
                num_correct += 1
        # print 'precision: %s' % (1.0 * c / len(l))

    container.append({'paper': scores[0]['doc_id'],
                      'predicted': len(prediction), 'correct': num_correct, 'hts': len(hts)})


def logistic_rescore(x, x0):
    k = 0.2
    return 1 / (1 + math.exp(k * (x - x0)))


def rerank(prediction, threshold, t2freq):
    if prediction[len(prediction) - 1][1] < threshold:
        prediction = sorted(prediction, cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1)
        # print prediction
        max_ht = 0
        for i in range(len(prediction)):
            so = prediction[i]
            if so[1] < threshold:
                max_ht = i

        ht_t2num = {}
        for i in range(len(prediction)):
            so = prediction[i]
            if len(so[2]) == 0:
                so[1] *= 0.7
                continue
            typed_ratio = 0
            if so[2] not in ht_t2num:
                ht_t2num[so[2]] = 0
            else:
                ht_t2num[so[2]] += 1
                typed_ratio = 1.0 * ht_t2num[so[2]] / max_ht
            so[1] = logistic_rescore(typed_ratio, t2freq[so[2]]) * so[1]
    return prediction


def naive_ne(t):
    t = t.lower().replace('\n', ' ')
    for r in HighLighter.get_brain_regions():
        if t.find(r.strip().lower()) >= 0:
            return True
    return False


def score_paper_threshold(score_file, container, out_file, hter, threshold):
    units = 5
    scores = utils.load_json_data(score_file)
    max_sid = int(scores[len(scores) - 1]['sid'])
    offset = int(1.0 * max_sid / units)

    anns = utils.load_json_data(scores[0]['doc_id'])
    hts = []
    sid2ann = {}
    sid2onto = {}
    ne_sids = []
    for ann in anns:
        if 'marked' in ann:
            hts.append(ann['sid'])
        sid2ann[ann['sid']] = ann
        if 'ncbo' in ann:
            matched_ontos = []
            for ncbo in ann['ncbo']:
                for name in pann.onto_name:
                    if name not in matched_ontos and ncbo['uri'].startswith(pann.onto_name[name]):
                        matched_ontos.append(name)
                    if name in matched_ontos:
                        break
            if len(matched_ontos) > 0:
                comb = '-'.join(sorted(matched_ontos))
                sid2onto[ann['sid']] = comb
        if naive_ne(ann['text']):
            ne_sids.append(ann['sid'])

    if len(hts) == 0:
        return

    prediction = []
    num_correct = 0
    r = 0
    t2freq = {}
    sp_ne_stat = HighLighter.get_sub_pred_ne_stat()

    precedent_sp = []
    precedent_threshold = 1

    sentence_level_details = []
    for i in range(len(scores)):
        score = scores[i]
        r = (i + 1) / offset
        score_ret = hter.score(score, region='r' + str(r))
        sent_type = '-'.join(sorted(score_ret['all_sps']))
        t2freq[sent_type] = 1 if sent_type not in t2freq else t2freq[sent_type] + 1

        onto2scores = HighLighter.get_onto_name_scores()
        onto_score = 0 if score['sid'] not in sid2onto else \
            0 if sid2onto[score['sid']] not in onto2scores \
                else onto2scores[sid2onto[score['sid']]]
        confidence = 0 if 'confidence' not in score['pattern'] else score['pattern']['confidence']
        sp_index = '-1' if 'sp_index' not in score['pattern'] else str(score['pattern']['sp_index'])
        sp_ne_freq = 0 if sp_index not in sp_ne_stat else sp_ne_stat[sp_index]

        other_ne = score['sid'] in ne_sids

        if (len(score_ret['sp']) > 0) \
                or (score_ret['cds'] + score_ret['nes'] > 0) \
                or other_ne \
                or onto_score > .2:
            s_sp = 0.0
            if len(score_ret['sp']) > 0:
                if len(score_ret['sp']) == 1:
                    for t in score_ret['sp']:
                        s_sp = score_ret['sp'][t]
                else:
                    type_score = []
                    for t in score_ret['sp']:
                        type_score.append([t, score_ret['sp'][t]])
                    type_score = sorted(type_score, cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1 )
                    s_sp = type_score[0][1]

            # s = (s_sp + score_ret['cds'] + score_ret['nes'])/3
            # s = 0.40 * s_sp + 0.2 * score_ret['cds'] + 0.4 * score_ret['nes']
            # s = 0.45 * s_sp + 0.4 * score_ret['cds'] + 0.25 * score_ret['nes']
            # s = 0.01 * s_sp + 0.29 * (score_ret['cds'] + (0.03 if other_ne else 0)) + 0.7 * score_ret['nes']
            # s = .1 * s_sp + .5 * score_ret['cds'] + .3 * ((0.3 if other_ne else 0) + score_ret['nes']) + .1 * onto_score
            s = 3 * score_ret['cds'] + 7 * (score_ret['nes'] + (0.05 if other_ne else 0)) + .01 * s_sp

            # F5: frequency of sub-pred patterns that associated with positive named entities
            if sp_ne_freq > 0:
                s += sp_ne_freq / 30 * s_sp

            # F4: sub-pred frequency scoring
            if int(sp_index) >= 0:
                sp_freq = hter.get_sub_pred()[int(sp_index)][1]
                s += 0 if sp_freq < 2 else sp_freq / 7 * s_sp

            # F1: neighbourhood boosting
            precedent_sp_freq = 0
            precedent_boost = 1
            for p_sp in precedent_sp:
                precedent_sp_freq += 0 if p_sp == 0 else 1
            precedent_boost = precedent_sp_freq + precedent_boost
            # s += 0 if s_sp == 0 or precedent_sp_freq == 0 else s_sp
            # s *= precedent_boost

            # F2: voting enhancement
            voted = 0.06
            if score_ret['nes'] > 0 or other_ne:
                voted += 1
            if score_ret['cds'] > 0:
                voted += 1
            s *= voted

            # F3: type regional boosting (spatial features)
            type_boost = .7 if r in [0, 1] else .3 if r in [2, 3] else 0.05
            region = 'r%s' % r
            sent_boost = HighLighter.get_sent_type_boost()
            if sent_type in sent_boost:
                type_boost = sent_boost[sent_type][region] if region in sent_boost[sent_type] else 0.001
            type_boost = math.pow(type_boost, 10.5)
            s *= type_boost

            prediction.append([score['sid'], s, sent_type])

            # push current sp info
            while len(precedent_sp) >= precedent_threshold:
                precedent_sp.pop()
            precedent_sp.append(s_sp)

            if score['sid'] in hts or s > threshold:
                sentence_level_details.append(
                    u'[{}]\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                        score['sid'],
                        'H' if score['sid'] in hts else '-',
                        'P' if s > threshold else '-',
                        sent_type,
                        '{}'.format(s, type_boost),
                        '{}/{}'.format(s_sp, confidence),
                        '{}'.format(score_ret['cds']),
                        score_ret['nes'],
                        anns[i]['text'].replace('\n', '').replace('\t', '')
                    )
                )
        else:
            if score['sid'] in hts:
                sentence_level_details.append(
                    u'[{}]\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                        score['sid'],
                        'H' if score['sid'] in hts else '-',
                        '-',
                        '-',
                        '-',
                        '-',
                        '-',
                        '-',
                        anns[i]['text'].replace('\n', '').replace('\t', '')
                    )
                )

    # calculate type stats
    # for t in t2freq:
    #     t2freq[t] = 1.0 * (t2freq[t] + 1) / max_sid

    # do reranking
    # for i in range(10):
    #     prediction = rerank(prediction, threshold, t2freq)

    prediction = sort_by_threshold(prediction, threshold,
                            cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1)

    for s in prediction:
        if s[0] in hts:
            num_correct += 1
    # print 'precision: {}, recall: {}'.format(1.0 * num_correct / len(prediction), 1.0 * num_correct / len(hts))

    container.append({'paper': scores[0]['doc_id'],
                      'predicted': len(prediction), 'correct': num_correct, 'hts': len(hts), 'max_sid': max_sid})
    return sentence_level_details


def get_sents_by_threshold(score_file, container, out_file, hter, threshold):
    units = 5
    scores = utils.load_json_data(score_file)
    max_sid = int(scores[len(scores) - 1]['sid'])
    offset = int(1.0 * max_sid / units)

    anns = utils.load_json_data(scores[0]['doc_id'])
    hts = []
    sid2ann = {}
    for ann in anns:
        if 'marked' in ann:
            hts.append(ann['sid'])
        sid2ann[ann['sid']] = ann

    if len(hts) == 0:
        return

    prediction = []
    num_correct = 0
    r = 0
    for i in range(len(scores)):
        score = scores[i]
        r = (i + 1) / offset
        score_ret = hter.score(score, region='r' + str(r))
        if len(score_ret['sp']) > 0 or (score_ret['cds'] + score_ret['nes'] > 0):
            container.append({'ht': score['sid'] in hts, 'text': sid2ann[score['sid']]['text'].replace('\n', '').replace('\t', '')})


def score_paper_by_type(score_file, container, out_file, hter, sent_type, threshold):
    units = 5
    scores = utils.load_json_data(score_file)
    max_sid = int(scores[len(scores) - 1]['sid'])
    offset = int(1.0 * max_sid / units)

    anns = utils.load_json_data(scores[0]['doc_id'])
    hts = []
    sid2ann = {}
    for ann in anns:
        if 'marked' in ann:
            hts.append(ann['sid'])
        sid2ann[ann['sid']] = ann

    if len(hts) == 0:
        return

    typed_hts = []
    prediction = []
    single_typed = []
    multi_typed = []
    others = []
    num_correct = 0
    r = 0
    for i in range(len(scores)):
        score = scores[i]
        r = (i + 1) / offset
        score_ret = hter.score(score, region='r' + str(r))
        if sent_type in score_ret['sp']:
            other_score = score_ret['cds'] + score_ret['nes']
            if score['sid'] in hts:
                typed_hts.append(score['sid'])
            if len(score_ret['sp']) == 1:
                single_typed.append([score['sid'], score_ret['sp'][sent_type] + other_score])
            else:
                multi_typed.append([score['sid'], score_ret['sp'][sent_type] + other_score])
            print u'{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(score['sid'] in typed_hts, '-'.join(score_ret['sp']),
                                                      score_ret['sp'][sent_type],
                                                      'r' + str(r),
                                                      score_ret['cds'], score_ret['nes'],
                                                      sid2ann[score['sid']]['text']
                                                      )

    if len(typed_hts) == 0:
        return # do not count if there is no such highlighted sentences

    prediction = sort_complement([], single_typed+multi_typed, threshold,
                            cmp=lambda p1, p2 : 1 if p2[1] > p1[1] else 0 if p2[1] == p1[1] else -1)

    for p in prediction:
        if p[0] in typed_hts:
            num_correct += 1

    container.append({'paper': scores[0]['doc_id'],
                      'predicted': len(prediction), 'correct': num_correct, 'hts': len(typed_hts)})


def represent_doc_as_vector(score_file, container, out_file, hter):
    units = 5
    scores = utils.load_json_data(score_file)
    max_sid = int(scores[len(scores) - 1]['sid'])
    offset = int(1.0 * max_sid / units)

    anns = utils.load_json_data(scores[0]['doc_id'])
    hts = []
    sid2ann = {}
    for ann in anns:
        if 'marked' in ann:
            hts.append(ann['sid'])
        sid2ann[ann['sid']] = ann

    if len(hts) == 0:
        return

    for i in range(len(scores)):
        score = scores[i]
        r = (i + 1) / offset
        vect, total = hter.get_vector(score, region='r' + str(r))
        if total > 0:
            container.append({'v': vect, 'ht': 1 if score['sid'] in hts else -1})


def pp_score_exp(container, out_file, hter, threshold):
    should = 0
    correct = 0
    predicted = 0
    total = 0

    # print 'precision\trecall\t#highlighted\t#predicted\tpaper'
    for p in container:
        should += p['hts']
        correct += p['correct']
        predicted += p['predicted']
        total += p['max_sid'] if 'max_sid' in p else 0
        # print '{:.2f}\t{:.2f}\t{}\t{}\t{}'.format(1.0 * p['correct'] / p['predicted'] if p['predicted'] > 0 else 0,
        #                               1.0 * p['correct'] / p['hts'],
        #                               p['hts'], p['predicted'], p['paper'])

    if predicted == 0:
        print '{}\t-\t-\t-'.format(threshold)
    else:
        precision = 1.0 * correct / predicted
        recall = 1.0 * correct / should
        # print 'precision\trecall\tF1'
        print '{}\t{}\t{}\t{}\t{}'.format(threshold, precision, recall,
                                          '-' if total == 0 else (1.0 * predicted - correct)/(total - correct),
                                          2 * precision * recall / (precision + recall))
        # utils.save_json_array(container, out_file)


def pp_get_sents(container, out_file, hter, threshold):
    hts = []
    nhts = []
    for s in container:
        if s['ht']:
            hts.append({'text': s['text']})
        else:
            nhts.append({'text': s['text']})
    ratio = 0.8
    ht_training = int(ratio*len(hts))
    nht_training = int(ratio*len(nhts))
    utils.save_json_array(hts[:ht_training], out_file + 'ht_training.json')
    utils.save_json_array(hts[ht_training:], out_file + 'ht_testing.json')
    utils.save_json_array(nhts[:nht_training], out_file + 'nht_training.json')
    utils.save_json_array(nhts[nht_training:], out_file + 'nht_testing.json')


def score_exp(score_files_path, out_file, threshold):
    ret_container = []
    hter = HighLighter.get_instance()
    utils.multi_thread_process_files(score_files_path, '', 10, score_paper_threshold,
                                     args=[ret_container, out_file, hter, threshold],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_score_exp)


def get_sent_exp(score_files_path, out_file, threshold):
    ret_container = []
    hter = HighLighter.get_instance()
    utils.multi_thread_process_files(score_files_path, '', 10, get_sents_by_threshold,
                                     args=[ret_container, out_file, hter, threshold],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_get_sents)


def pp_score_typed_exp(container, out_file, hter, sent_type, threshold):
    should = 0
    correct = 0
    predicted = 0

    # print 'precision\trecall\t#highlighted\t#predicted\tpaper'
    # print 'total papers: %s' % len(container)
    for p in container:
        should += p['hts']
        correct += p['correct']
        predicted += p['predicted']
        # print '{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}'.format(0 if p['predicted'] == 0 else 1.0 * p['correct'] / p['predicted'],
        #                               1.0 * p['correct'] / p['hts'],
        #                               p['hts'], p['predicted'], p['paper'])

    precision = 1.0 * correct / predicted
    recall = 1.0 * correct / should
    # print 'precision\trecall\tF1'
    print '{}.\t{}\t{}\t{}'.format(threshold, precision, recall, 2 * precision * recall / (precision + recall))
    # utils.save_json_array(container, out_file)


def pp_vectorise(container, out_file, hter):
    dims = []
    # for k in hter.get_score_cd():
    #     dims.append(k)
    for k in hter.get_score_nes():
        dims.append(k)
    for k in hter.get_score_sp():
        dims.append(k)

    hts = []
    nhts = []
    for v in container:
        if v['ht'] == 1:
            hts.append(v)
        else:
            nhts.append(v)

    learning_ratio = 0.8
    b_ht = int(learning_ratio * len(hts))
    b_nht = int(learning_ratio * len(nhts))

    s = '{} {} {}\n'.format(b_ht + b_nht, len(dims), 2)
    for v in hts[:b_ht]:
        r = ''
        for d in dims:
            r += '{}'.format('0' if d not in v['v'] else v['v'][d])
            r += ' '
        r = r.strip()
        r += '\n'
        r += '1 0\n'
        s += r
    for v in nhts[:b_nht]:
        r = ''
        for d in dims:
            r += '{}'.format('0' if d not in v['v'] else v['v'][d])
            r += ' '
        r = r.strip()
        r += '\n'
        r += '0 1\n'
        s += r
    utils.save_text_file(s, out_file)
    utils.save_json_array({'dims': dims, 'data': hts[b_ht:] + nhts[b_nht:]}, out_file + '_testing')


def score_exp_typed(score_files_path, sent_type, out_file, threshold):
    ret_container = []
    hter = HighLighter.get_instance()
    utils.multi_thread_process_files(score_files_path, '', 10, score_paper_by_type,
                                     args=[ret_container, out_file, hter, sent_type, threshold],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_score_typed_exp)


def vectorise_export(score_files_path, out_file):
    ret_container = []
    hter = HighLighter.get_instance()
    utils.multi_thread_process_files(score_files_path, '', 10, represent_doc_as_vector,
                                     args=[ret_container, out_file, hter],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_vectorise)


def dump_file_results(files, out_file):
    ht = HighLighter.get_instance()
    ctn = []
    s = 'sid\thighlighted\tpredicted\ttype\toverall score\tsub-pred score/confidence\tCD Score\tNE Score\ttext\n'
    for f in files:
        s += '\n\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            '**','**','**','**','**','**','**','**', f)
        s += '\n'.join(score_paper_threshold(f, ctn, '', ht, 0.003))
        print '%s done' % f
    utils.save_text_file(s, out_file)


def random_pick_files():
    test_files = [
        './20-test-papers/summaries/10561930_annotated_ann_scores.json',
        './20-test-papers/summaries/10791835_annotated_ann_scores.json',
        './20-test-papers/summaries/11277563_annotated_ann_scores.json',
        './20-test-papers/summaries/12124484_annotated_ann_scores.json',
        './20-test-papers/summaries/12673603_annotated_ann_scores.json',
        './20-test-papers/summaries/15178945_annotated_ann_scores.json',
        './20-test-papers/summaries/15377698_annotated_ann_scores.json',
        './20-test-papers/summaries/15645532_annotated_ann_scores.json',
        './20-test-papers/summaries/15661114_annotated_ann_scores.json',
        './20-test-papers/summaries/15942197_annotated_ann_scores.json',
    ]

    training_files = [
        './summaries/Bartova et al., (2010) - Correlation between substantia nigra features detected by sonography and PD_annotated_ann_scores.json',
        './summaries/Bilello et al., (2015) - Correlating cognitive decline with white matter lesions and atrophy in AD_annotated_ann_scores.json',
        './summaries/Forster et al., (2011) - Effects of a 6 month cognitive intervention program on brain metabolism in aMCI and mild AD_annotated_ann_scores.json',
        './summaries/Giannakopoulous et al., (2000) - Neural substrates of spatial and temporal disorientation in AD_annotated_ann_scores.json',
        './summaries/Sunwoo et al., (2013) - Thalamic volume and related visual recognition are associated with FOG in PD_annotated_ann_scores.json',
        './summaries/Ibarretxe-Bilbao et al., (2009) - Differential progression of brain atrophy in PD with and without VH_annotated_ann_scores.json',
        './summaries/Iranzo et al., (2002) - Sleep symptoms and polysomnographic architecture in advanced PD after chronic bilateral STN stimulation_annotated_ann_scores.json',
        './summaries/Lawrence et al., (2003) - Multiple neuronal networks mediate sustained attention_annotated_ann_scores.json',
        './summaries/Nee et al., (2014) - Prefrontal cortex organisation. Dissociating effects of temporal abstraction, relational abstraction and integration with fMRI_annotated_ann_scores.json',
        './summaries/Tan et al., (2015) - Pain in PD_annotated_ann_scores.json',
    ]
    dump_file_results(test_files, './results/sample_test_files.tsv')

if __name__ == "__main__":
    # visualise_result('./training/test/non_hts_scores.json', './training/test/hts_scores.json')
    # summarise_all_papers('./30-test-papers/', './30-test-papers/summaries/')
    # summ('./anns_v2/Ahn et al., (2011) - The cortical neuroanatomy of neuropsychological deficits in MCI and AD_annotated_ann.json',
    #     HighLighter.get_instance(),
    #      './summaries/')
    # sum, scores = ht.summarise([u'This is in agreement with findings from volumetric magnetic resonance ima- ging (MRI) studies which to date have provided clear evidence that hippocampal atrophy is a valuable method to support the clinical diagnosis of early AD [14, 22, 27, 28, 37].'])'./results/sample_test_files.txt')
    #             ctn, '', ht, {'goal': 2, 'findings': 0.02, 'method': 0.05})
    # score_paper_threshold('./20-test-papers/summaries/12673603_annotated_ann_scores.json',
    #                     [], '', HighLighter.get_instance(), 0.003)
    # print ctn
    # for i in np.arange(0, 1, 0.0005):
    #     score_exp('./20-test-papers/summaries/', './training/auto_ht_results.json', i)
    # score_exp('./30-test-papers/summaries/', './training/auto_ht_results.json', 0.0035)
    # score_exp_typed('./summaries/', 'goal', './training/auto_ht_results.json', 2)
    # get_sent_exp('./summaries/', './training/second_it_', 0.2)
    # vectorise_export('./summaries/', './training/vect_sents.json')
    random_pick_files()
