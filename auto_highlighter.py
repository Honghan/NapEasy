import json
import ann_analysor as aa
import codecs

res_file_cd = './resources/cardinal_Noun_patterns.txt'
res_file_ne = './resources/named_entities.txt'
res_file_sp = './resources/sub_pred.txt'


class HighLighter:

    def __init__(self, ne_res, cardinal_noun_res, sub_pred_res):
        self.ne = ne_res
        self.card = cardinal_noun_res
        self.sp = sub_pred_res
        self.stanford_parser = aa.create_stanford_parser_inst()

    def score(self, sent_text):
        scores = {'cd': 0, 'ne': 0, 'sp': 0}
        cd_nouns = {}
        named_entities = {}
        aa.extract_cd_nouns_nes(sent_text, cd_nouns, named_entities)
        for cdn in cd_nouns:
            scores['cd'] += 0 if cdn not in self.card else self.card[cdn]
        for ne in named_entities:
            scores['ne'] += 0 if ne not in self.ne else self.ne[ne]
        try:
            sub, pred = aa.analysis_sentence_text(self.stanford_parser, sent_text)
            sp = aa.SubjectPredicate(sub, pred)
            scores['sp'] = 0 if sp not in self.sp else self.sp[sp]
        except:
            print(u'failed parsing sentences for {0}'.format(sent_text))

        scores['total'] = scores['cd'] + scores['ne'] + scores['sp']
        return scores

    # get the instance of this class
    @staticmethod
    def get_instance():
        ne, cd, sp = load_resources(res_file_ne, res_file_cd, res_file_sp)
        return HighLighter(ne, cd, sp)


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
        sps[aa.SubjectPredicate(sp_raw[i][0]['s'], sp_raw[i][0]['p'])] = sp_raw[i][1]
    return sps


def load_resources(ne_file, cd_file, sp_file):
    ne = read_text_res(ne_file)
    cd = read_text_res(cd_file)
    sp = read_sub_pred_file(sp_file)
    return ne, cd, sp


def do_highlight(test_file):
    her = HighLighter.get_instance()
    data = None
    with codecs.open(test_file, encoding='utf-8') as rf:
        data = json.load(rf)
    return [her.score(item['text'])['total'] for item in data]


def do_compare():
    score1 = do_highlight('./training/non_hts0.json')
    score2 = do_highlight('./training/hts0.json')
    aa.plot_two_sets_data(score1, score2)

do_compare()
