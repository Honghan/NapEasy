import json
import ann_analysor as aa
import codecs
import ann_utils as utils
from os.path import split, join, isfile
import threading

res_file_cd = './resources/cardinal_Noun_patterns.txt'  #'./training/patterns/cardinal_noun.txt'
res_file_ne = './resources/named_entities.txt' # './training/patterns/named_entities.txt'
res_file_sp = './resources/sub_pred.txt' # './training/patterns/sub_pred.json'
res_file_spcat = './resources/sub_pred_categories.json'

parser_lock = threading.RLock()

class HighLighter:

    def __init__(self, parser, ne_res, cardinal_noun_res, sub_pred_res, sub_pred_cats=None):
        self.ne = ne_res
        self.card = cardinal_noun_res
        self.sp = sub_pred_res
        self.sp_cats = sub_pred_cats
        self.stanford_parser = parser
        # print('loading stanford parser...')
        # self.stanford_parser = aa.create_stanford_parser_inst()
        # print('stanford parser loaded')

    def score(self, sent_text, doc_id=None, sid=None, container=None):
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
                with parser_lock:
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
                self.score(sent, doc_id=src, sid=sid)
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
    def pick_top_k(sents, k):
        if len(sents) == 0:
            return sents

        sorted_sents = sorted(sents, cmp=lambda s1, s2: s2[1]['total'] - s1[1]['total'])
        return sorted_sents[:k] if 'sid' not in sorted_sents[0][1] \
            else sorted(sorted_sents[:k], cmp=lambda s1, s2: int(s1[1]['sid']) - int(s2[1]['sid']))

    # get the instance of this class
    @staticmethod
    def get_instance():
        parser = aa.create_stanford_parser_inst()
        ne, cd, sp, cats = load_resources(res_file_ne, res_file_cd, res_file_sp, res_file_spcat)
        return HighLighter(parser, ne, cd, sp, cats)


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


def load_resources(ne_file, cd_file, sp_file, sp_cat_file=None):
    ne = read_text_res(ne_file)
    cd = read_text_res(cd_file)
    sp = read_sub_pred_file(sp_file)
    sp_cats = None if sp_cat_file is None else utils.load_json_data(sp_cat_file)
    return ne, cd, sp, sp_cats


def score_sentence(item, her, container, out_file=None):
    her.score(item['text'], doc_id=item['src'], sid=item['sid'], container=container)


def do_highlight(test_file):
    print('initialising highlighter instance...')
    her = HighLighter.get_instance()
    print('highlighter instance initialised')
    data = None
    with codecs.open(test_file, encoding='utf-8') as rf:
        data = json.load(rf)
    scores = []
    out_file = test_file[:test_file.rfind('.')] + "_scores.json"
    print('multithreading...')
    utils.multi_thread_tasking(data, 5, score_sentence, args=[her, scores, out_file],
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


def summ(ann_file, highlighter, out_path):
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
    her = HighLighter.get_instance()
    utils.multi_thread_process_files(ann_path, '', 6, summ,
                                     args=[her, summ_path],
                                     file_filter_func=lambda f: f.endswith('_ann.json'))

if __name__ == "__main__":
    # visualise_result('./training/test/non_hts_scores.json', './training/test/hts_scores.json')
    summarise_all_papers('./anns_v2/', './summaries/')
    # summ('./anns_v2/Ahn et al., (2011) - The cortical neuroanatomy of neuropsychological deficits in MCI and AD_annotated_ann.json',
    #     HighLighter.get_instance(),
    #      './summaries/')
    # ht = HighLighter.get_instance()
    # sum, scores = ht.summarise([u'This is in agreement with findings from volumetric magnetic resonance ima- ging (MRI) studies which to date have provided clear evidence that hippocampal atrophy is a valuable method to support the clinical diagnosis of early AD [14, 22, 27, 28, 37].'])
    # print scores
