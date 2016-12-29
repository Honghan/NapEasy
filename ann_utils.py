from os import listdir
from os.path import isfile, join, split
import Queue
import threading
import json
import codecs
import nltk
import requests
import re
from pyquery import PyQuery as pq
from nltk.corpus import wordnet as wn
from nltk.data import load
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import auto_highlighter as ah
import ann_analysor as aa
import traceback
import multiprocessing

# ncbi etuils url
ncbi_service_url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?' \
                   'db=pubmed&term={}&field=title&retmode=json'
ncbi_pubmed_url = 'https://www.ncbi.nlm.nih.gov/pubmed/?term={}'
ncbi_host = 'https://www.ncbi.nlm.nih.gov'

relation_pos_list = [
                'RB', 'RBR', 'RBS',
                'JJ', 'JJR', 'JJS',
                # 'NN', 'NNS', 'NNP', 'NNPS',
                'VB', 'VBD', 'VBN', 'VBG', 'VBP', 'VBZ']

# list files in a folder and put them in to a queue for multi-threading processing
def multi_thread_process_files(dir_path, file_extension, num_threads, process_func,
                               proc_desc='processed', args=None, multi=None,
                               file_filter_func=None, callback_func=None,
                               thread_wise_objs=None):
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    num_pdfs = 0
    files = None if multi is None else []
    lst = []
    for f in onlyfiles:
        if f.endswith('.' + file_extension) if file_filter_func is None \
                else file_filter_func(f):
            if multi is None:
                lst.append(join(dir_path, f))
            else:
                files.append(join(dir_path, f))
                if len(files) >= multi:
                    lst.append(files)
                    files = []
            num_pdfs += 1
    if files is not None and len(files) > 0:
        lst.append(files)
    multi_thread_tasking(lst, num_threads, process_func, proc_desc, args, multi, file_filter_func,
                         callback_func,
                         thread_wise_objs=thread_wise_objs)


def multi_thread_tasking(lst, num_threads, process_func,
                               proc_desc='processed', args=None, multi=None,
                               file_filter_func=None, callback_func=None, thread_wise_objs=None):
    num_pdfs = len(lst)
    pdf_queque = Queue.Queue(num_pdfs)
    # print('putting list into queue...')
    for item in lst:
        pdf_queque.put_nowait(item)
    thread_num = min(num_pdfs, num_threads)
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, pdf_queque)
    # print('queue filled, threading...')
    for i in range(thread_num):
        tarr = arr[:]
        thread_obj = None
        if thread_wise_objs is not None and isinstance(thread_wise_objs, list):
            thread_obj = thread_wise_objs[i]
        tarr.insert(0, thread_obj)
        t = threading.Thread(target=multi_thread_do, args=tuple(tarr))
        t.daemon = True
        t.start()

    # print('waiting jobs to finish')
    pdf_queque.join()
    # print('{0} files {1}'.format(num_pdfs, proc_desc))
    if callback_func is not None:
        callback_func(*tuple(args))


def multi_thread_do(thread_obj, q, func, *args):
    while True:
        p = q.get()
        try:
            if thread_obj is not None:
                func(thread_obj, p, *args)
            else:
                func(p, *args)
        except Exception, e:
            print u'error doing {0} on {1} \n{2}'.format(func, p, str(e))
            traceback.print_exc()
        q.task_done()


# begin: multiple processing functions
def multi_process_tasking(lst, num_processes, process_func,
                          args=None, callback_func=None, process_wise_objs=None):
    num_items = len(lst)
    item_queue = multiprocessing.JoinableQueue(num_items)
    # print('putting list into queue...')
    for item in lst:
        item_queue.put_nowait(item)
    thread_num = min(num_items, num_processes)
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, item_queue)
    # print('queue filled, threading...')
    processes = []
    for i in range(thread_num):
        tarr = arr[:]
        process_obj = None
        if process_wise_objs is not None and isinstance(process_wise_objs, list):
            process_obj = process_wise_objs[i]
        tarr.insert(0, process_obj)
        t = multiprocessing.Process(target=multi_process_do, args=tuple(tarr))
        t.daemon = True
        t.start()
        processes.append(t)

    # print('waiting jobs to finish')
    item_queue.join()
    for t in processes:
        t.terminate()
    # print('{0} files {1}'.format(num_pdfs, proc_desc))
    if callback_func is not None:
        callback_func(*tuple(args))


def multi_processing_process_files(dir_path, file_extension, num_processes, process_func,
                                   proc_desc='processed', args=None, multi=None,
                                   file_filter_func=None, callback_func=None,
                                   process_wise_objs=None):
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    num_pdfs = 0
    files = None if multi is None else []
    lst = []
    for f in onlyfiles:
        if f.endswith('.' + file_extension) if file_filter_func is None \
                else file_filter_func(f):
            if multi is None:
                lst.append(join(dir_path, f))
            else:
                files.append(join(dir_path, f))
                if len(files) >= multi:
                    lst.append(files)
                    files = []
            num_pdfs += 1
    if files is not None and len(files) > 0:
        lst.append(files)
    multi_process_tasking(lst, num_processes, process_func, args,
                          callback_func=callback_func, process_wise_objs=process_wise_objs)


def multi_process_do(process_obj, q, func, *args):
    while True:
        p = q.get()
        try:
            if process_obj is not None:
                func(process_obj, p, *args)
            else:
                func(p, *args)
        except Exception, e:
            print u'error doing {0} on {1} \n{2}'.format(func, p, str(e))
            traceback.print_exc()
        q.task_done()
# end: multiple processing functions


def filter_path_file(dir_path, file_extension=None, file_filter_func=None):
    return [f for f in listdir(dir_path)
            if isfile(join(dir_path, f))
            and (f.endswith('.' + file_extension) if file_filter_func is None else file_filter_func(f))]


def relation_patterns(s):
    text = nltk.word_tokenize(s)
    pr = nltk.pos_tag(text)
    picked = []
    for p in pr:
        if p[1] in relation_pos_list:
            picked.append(p[0])
    return ' '.join(picked)


def convert_ann_for_training(ann_file, non_hts, hts, out_path):
    anns = None
    with codecs.open(ann_file, encoding='utf-8') as rf:
        anns = json.load(rf)
    p, fn = split(ann_file)
    for ann in anns:
        co = {
                # 'src': fn,
                # 'sid': ann['sid'],
                'text': ann['text'],
                # 'struct': '' if 'struct' not in ann else ann['struct'],
                # 'sapienta': '' if 'CoreSc' not in ann else ann['CoreSc'],
                # 'entities': '' if 'ncbo' not in ann else ' '.join( list(set([a['annotation']['text'].lower() for a in ann['ncbo']])) )
              }
        if 'marked' in ann:
            # co['marked'] = ann['marked']
            hts.append(co)
        else:
            # co['marked'] = ''
            non_hts.append(co)
    print('{} done'.format(ann_file))


def ann_to_training(ann_file_path, output_path):
    non_hts = []
    hts = []
    multi_thread_process_files(ann_file_path, '', 2, convert_ann_for_training,
                               args=[non_hts, hts, output_path], file_filter_func=lambda f: f.endswith('_ann.json'),
                               callback_func=save_sentences)


def save_json_array(lst, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as wf:
        json.dump(lst, wf, encoding='utf-8')


def save_text_file(text, file_path):
    with codecs.open(file_path, 'w', encoding='utf-8') as wf:
        wf.write(text)


def append_text_file(text, file_path):
    with codecs.open(file_path, 'a', encoding='utf-8') as wf:
        wf.write(text)


def load_json_data(file_path):
    data = None
    with codecs.open(file_path, encoding='utf-8') as rf:
        data = json.load(rf, encoding='utf-8')
    return data


def load_text_file(file_path):
    data = []
    with codecs.open(file_path, encoding='utf-8') as rf:
        data = rf.readlines()
    return data


def save_sentences(non_hts, hts, output_path):
    training_testing_ratio = 1
    total_num = min(len(hts), len(non_hts))
    trainin_len = int(training_testing_ratio * total_num)
    save_json_array(non_hts, join(output_path, 'non_hts.json'))
    save_json_array(hts, join(output_path, 'hts.json'))

    # save_json_array(non_hts[:trainin_len], join(output_path, 'non_hts.json'))
    # save_json_array(hts[:trainin_len], join(output_path, 'hts.json'))
    # save_json_array(non_hts[trainin_len:total_num], join(output_path + "/test", 'non_hts.json'))
    # save_json_array(hts[trainin_len:total_num], join(output_path + "/test", 'hts.json'))

    # split training data into equally sized groups
    # num_group = 3
    # for i in range(num_group):
    #     s = i * trainin_len / num_group
    #     e = min((i+1) * trainin_len / num_group, trainin_len)
    #     ts = trainin_len + i * (total_num - trainin_len) / num_group
    #     te = min(trainin_len + (i + 1) * (total_num - trainin_len) / num_group, total_num)
    #     save_json_array(non_hts[s:e], join(output_path, 'non_hts' + str(i) +'.json'))
    #     save_json_array(hts[s:e], join(output_path, 'hts' + str(i) +'.json'))
    #     save_json_array(non_hts[ts:te], join(output_path + "/test", 'non_hts' + str(i) +'.json'))
    #     save_json_array(hts[ts:te], join(output_path + "/test", 'hts' + str(i) +'.json'))

    print('all done [training size: {0}, testing size: {1}]'.format(trainin_len, total_num - trainin_len))


def add_pmcid_to_sum(sum_file_path):
    summ = load_json_data(sum_file_path)
    # if 'PMID' in summ:
    #     return
    p, fn = split(sum_file_path)
    m = re.match(r'[^()]+\(\d+\) \- (.+)_annotated_ann\.sum', fn)
    pmcid = None
    journal = None
    cnt = None
    if m is not None:
        # ret = json.loads(requests.get(ncbi_service_url.format(m.group(1))).content)
        cnt = requests.get(ncbi_pubmed_url.format(m.group(1))).content
    else:
        m = re.match(r'(\d+)_annotated_ann\.sum', fn)
        if m is not None:
            pmcid = m.group(1)
            cnt = requests.get(ncbi_pubmed_url.format(m.group(1))).content
    if cnt is not None:
        doc = pq(cnt)
        # check whether it is a list of search results
        results = doc(".result_count.left").eq(0)
        if results.html() is not None:
            dom_str = doc(".rslt > .title").eq(0)
            if dom_str is not None and dom_str.html() is not None:
                pmcid = extract_pubmed(dom_str.html())

            j_elem = doc(".jrnl").eq(0)
            if j_elem is not None and j_elem.html() is not None:
                journal = j_elem.html()
        else:
            if pmcid is None:
                dom_str = doc(".rprtid").eq(0)
                if dom_str is not None and dom_str.html() is not None:
                    pmcid = extract_pubmed(dom_str.html())
            j_elem = doc(".cit").eq(0)
            if j_elem is not None and j_elem.html() is not None:
                m1 = re.findall(r'alterm="([^"]*)"', str(j_elem.html()))
                if m1 is not None:
                    if len(m1) > 0:
                        journal = m1[0][0:len(m1[0])-1]
        # if p is not None and len(p.strip()) > 0:

        # if ret is None or len(ret['esearchresult']['idlist']) == 0:
        #     print 'no pmc id found for {}'.format(sum_file_path)
        # else:
        #     pmcid = ret['esearchresult']['idlist']
    summ['PMID'] = pmcid
    if journal is not None:
        journal = pq(journal).text()
    summ['journal'] = journal
    print pmcid, journal, sum_file_path
    save_json_array(summ, sum_file_path)


def extract_pubmed(html_str):
    pmcid = None
    m1 = re.findall(u'("/pubmed/(\d+)")|(PMID:</dt>.+XInclude">(\d+)</dd>)', html_str)
    if m1 is not None:
        if len(m1[0][1]) > 0:
            pmcid = m1[0][1]
        elif len(m1[0][3]) > 0:
            pmcid = m1[0][3]
    return pmcid


def process_pmcids(sum_folder):
    multi_thread_process_files(sum_folder, 'sum', 3, add_pmcid_to_sum)


def check_score_func_sids(score_file, container):
    scores = load_json_data(score_file)
    anns = load_json_data(scores[0]['doc_id'])
    prob_pairs = []
    for i in range(len(scores)):
        if scores[i]['sid'] != anns[i]['sid']:
            scores[i]['sid'] = anns[i]['sid']
            prob_pairs.append({'score_sid': scores[i]['sid'], 'ann_sid': anns[i]['sid']})
    if len(prob_pairs) > 0:
        container.append({'f':score_file, 'p':prob_pairs})
        save_json_array(scores, score_file)


def sid_check_cb(probs):
    print json.dumps(probs)


def check_all_scores_sids(sum_folder):
    probs = []
    multi_thread_process_files(sum_folder, 'json', 3, check_score_func_sids,
                               args=[probs],
                               callback_func=sid_check_cb)


_treebank_word_tokenize = TreebankWordTokenizer().tokenize
# stemmer instance
porter = nltk.PorterStemmer()
stop_words = stopwords.words('english')

def word_tokenize(text, language='english'):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    return [token for sent in sent_tokenize(text, language)
            for token in _treebank_word_tokenize(sent)]


# Standard sentence tokenizer.
def sent_tokenize(text, language='english'):
    """
    Return a sentence-tokenized copy of *text*,
    using NLTK's recommended sentence tokenizer
    (currently :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into sentences
    :param language: the model name in the Punkt corpus
    """
    tokenizer = load('tokenizers/punkt/{0}.pickle'.format(language))
    return tokenizer.tokenize(text)


def token_text(text, pos='n'):
    words = []
    for t in word_tokenize(text):
        if len(t) > 1 and t not in stop_words:
            words.append(t)
    return [WordNetLemmatizer().lemmatize(t.lower(), pos) for t in words]


def word_similarity(w1, w2, pos=wn.NOUN):
    if len(w1) == 0 or len(w2) == 0:
        return 0, None
    set1 = wn.synsets(w1, pos=pos)
    set2 = wn.synsets(w2, pos=pos)
    max = 0
    p = []
    for s1 in set1:
        for s2 in set2:
            sim = s1.path_similarity(s2)
            if sim > max:
                max = sim
                p = [s1, s2]
    return max, p


def phrase_similarity(p1, p2, pos='n'):
    if p1.strip() == p2.strip():
        return 1
    arr1 = token_text(p1, pos)
    arr2 = token_text(p2, pos)
    m = 0
    for t1 in arr1:
        for t2 in arr2:
            sim, p = word_similarity(t1, t2, wn.NOUN if 'n' == pos else wn.VERB)
            if sim > m:
                m = sim
    return m


def match_sp_type(sp_patterns, sp_cats, subs, preds, paper_id=None, sid=None, result_container=None):
    idx2score = {}
    p2score = {}
    for p in sp_cats:
        m = 0
        m_idx = -1
        for idx in sp_cats[p]:
            s = 0
            if idx in idx2score:
                s = idx2score[idx]
            else:
                sp = sp_patterns[idx][0]
                s_score = phrase_similarity(' '.join(subs) if subs is not None else '',
                                            ' '.join(sp['s'] if sp['s'] is not None else ''))
                p_score = phrase_similarity(' '.join(preds) if preds is not None else '',
                                            ' '.join(sp['p'] if sp['p'] is not None else ''), 'v')
                s = min(s_score, p_score)
                idx2score[idx] = s
            if s > m:
                m = s
                m_idx = idx
        p2score[p] = {'m': m, 'idx': m_idx}
    mp = ''
    m = 0
    m_idx = -1
    for p in p2score:
        if p2score[p]['m'] > m:
            m = p2score[p]['m']
            mp = p
            m_idx = p2score[p]['idx']
    if m < 0.5:
        return None, None, None
    if result_container is not None:
        result_container.put({'sim': m, 'pattern': mp, 'index': m_idx, 'sid': sid, 'paper_id': paper_id})
    return m, mp, m_idx


def semantic_fix_scores(score_file, sp_patterns, sp_cats):
    print 'working on %s...' % score_file
    scores = load_json_data(score_file)
    for s in scores:
        p = s['pattern']
        if 'sp_index' in p and p['sp_index'] == -1 and 'sub' in p:
            m, mp, m_idx = match_sp_type(sp_patterns, sp_cats, p['sub'], p['pred'])
            if m is not None:
                p['sp_index'] = m_idx
                print s['sid'], p['sub'], p['pred'], mp
    save_json_array(scores, score_file)
    print '%s done.' % score_file


def semantic_fix_scores_confidence(score_file, sp_patterns, sp_cats, hter, score_path):
    print 'working on %s...' % score_file
    scores = load_json_data(score_file)
    for s in scores:
        p = s['pattern']
        if 'sp_index' in p and p['sp_index'] == -1:
            sp = aa.SubjectPredicate(p['sub'], p['pred'])
            if sp in hter.sp:
                p['sp_index'] = hter.sp[sp]['index']
                p['confidence'] = 2
                print sp, p['sp_index']
            else:
                m, mp, m_idx = match_sp_type(sp_patterns, sp_cats, p['sub'], p['pred'])
                if m is not None:
                    p['sp_index'] = m_idx
                    p['confidence'] = m
                    print s['sid'], p['sub'], p['pred'], mp
    save_json_array(scores, score_file)
    print '%s done.' % score_file


def semantic_fix_all_scores(socre_folder_path, cb=None):
    hter = ah.HighLighter.get_instance()
    sp_patterns = load_json_data('./resources/sub_pred.txt')
    sp_cats = load_json_data('./resources/sub_pred_categories.json')
    # multi_thread_process_files(socre_folder_path, '', 1, semantic_fix_scores_confidence,
    #                                  args=[sp_patterns, sp_cats, hter, socre_folder_path],
    #                                  file_filter_func=lambda fn: fn.endswith('_scores.json'),
    #                            callback_func=cb)
    multi_processing_process_files(socre_folder_path, '', 5, semantic_fix_scores_confidence,
                                   args=[sp_patterns, sp_cats, hter, socre_folder_path],
                                   file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                   callback_func=cb)


# multiple-processing the semantic fixing on a batch of sentences
def semantic_fix_worker(job, sp_patterns, sp_cats, container, score_folder_path, cb):
    match_sp_type(sp_patterns, sp_cats, job['sub'], job['pred'],
                  paper_id=job['paper_id'], sid=job['sid'], result_container=container)


def semantic_fix_finish(sp_patterns, sp_cats, container, score_folder_path, cb):
    paper_to_matched = {}
    while not container.empty():
        m = container.get_nowait()
        if m['paper_id'] not in paper_to_matched:
            paper_to_matched[m['paper_id']] = {}
        paper_to_matched[m['paper_id']][m['sid']] = m

    for score_file in paper_to_matched:
        print 'putting results back to %s...' % score_file
        scores = load_json_data(score_file)
        matched_patterns = paper_to_matched[score_file]
        for s in scores:
            if s['sid'] in matched_patterns:
                m = matched_patterns[s['sid']]
                p = s['pattern']
                p['sp_index'] = m['index']
                p['confidence'] = m['sim']
                print s['sid'], p['sub'], p['pred'], m['pattern']
        save_json_array(scores, score_file)
        print '%s done.' % score_file
    print 'all semantically fixed'
    if cb is not None:
        cb(score_folder_path)


def multi_processing_semantic_fix_all_scores(score_folder_path, cb=None):
    hter = ah.HighLighter.get_instance()
    sp_patterns = load_json_data('./resources/sub_pred.txt')
    sp_cats = load_json_data('./resources/sub_pred_categories.json')
    sentence_job_list = []
    files = filter_path_file(score_folder_path, file_filter_func=lambda fn: fn.endswith('_scores.json'))
    job_size = 0
    for f in files:
        score_file = join(score_folder_path, f)
        print 'pulling from %s...' % score_file
        scores = load_json_data(score_file)
        for s in scores:
            p = s['pattern']
            if 'sp_index' in p:
                sp = aa.SubjectPredicate(p['sub'], p['pred'])
                if sp in hter.sp:
                    p['sp_index'] = hter.sp[sp]['index']
                    p['confidence'] = 2
                    print sp, p['sp_index']
                elif 'sp_index' in p and p['sp_index'] == -1:
                    sentence_job_list.append({'paper_id':score_file, 'sid': s['sid'], 'sub': p['sub'], 'pred':p['pred']})
                    job_size += 1
        print '%s pulled.' % score_file
        save_json_array(scores, score_file)
    results = multiprocessing.Queue(job_size)
    multi_process_tasking(sentence_job_list, 3, semantic_fix_worker,
                          args=[sp_patterns, sp_cats, results, score_folder_path, cb],
                          callback_func=semantic_fix_finish)
# end of multiple-processing the semantic fixing on a batch of sentences


def main():
    # ann_to_training('./local_exp/anns_v2', './training')
    # sents = [
    #     'The control group was comprised of 15 elderly community dwelling individuals of comparable age and educational background',
    #     'This resulted in data of 172 participants to be included in the present study.'
    # ]
    # relation_patterns(sents[0])
    # add_pmcid_to_sum('./20-test-papers/summaries/10561930_annotated_ann.sum')
    # process_pmcids('./20-test-papers/summaries/')
    # check_score_func_sids('./summaries/Altug et al., (2011) - The influence of subthalamic nucleus DBS on daily living activities in PD_annotated_ann_scores.json')
    # check_all_scores_sids('./summaries/')
    # phrase_similarity('were included', 'were made', 'v')
    # sp_patterns = load_json_data('./resources/sub_pred.txt')
    # sp_cats = load_json_data('./resources/sub_pred_categories.json')
    # print match_sp_type(sp_patterns, sp_cats, ['conclusions'], ['drawn'])
    # semantic_fix_scores('./30-test-papers/summaries/10561930_annotated_ann_scores.json', sp_patterns, sp_cats)
    # semantic_fix_all_scores('./local_exp/42-extra-papers/summaries/')
    multi_processing_semantic_fix_all_scores('/Users/jackey.wu/Documents/working/KCL/psychometricTests_neuroanatomy/local_exp/test_mp_sem_fix')

if __name__ == "__main__":
    main()
