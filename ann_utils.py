from os import listdir
from os.path import isfile, join, split
import Queue
import threading
import json
import codecs
import nltk

relation_pos_list = [
                'RB', 'RBR', 'RBS',
                'JJ', 'JJR', 'JJS',
                # 'NN', 'NNS', 'NNP', 'NNPS',
                'VB', 'VBD', 'VBN', 'VBG', 'VBP', 'VBZ']

# list files in a folder and put them in to a queue for multi-threading processing
def multi_thread_process_files(dir_path, file_extension, num_threads, process_func,
                               proc_desc='processed', args=None, multi=None,
                               file_filter_func=None, callback_func=None):
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
    multi_thread_tasking(lst, num_threads, process_func, proc_desc, args, multi, file_filter_func, callback_func)


def multi_thread_tasking(lst, num_threads, process_func,
                               proc_desc='processed', args=None, multi=None,
                               file_filter_func=None, callback_func=None):
    num_pdfs = len(lst)
    pdf_queque = Queue.Queue(num_pdfs)
    print('putting list into queue...')
    for item in lst:
        pdf_queque.put_nowait(item)
    thread_num = min(num_pdfs, num_threads)
    arr = [process_func] if args is None else [process_func] + args
    arr.insert(0, pdf_queque)
    print('queue filled, threading...')
    for i in range(thread_num):
        t = threading.Thread(target=multi_thread_do, args=tuple(arr))
        t.daemon = True
        t.start()

    print('waiting jobs to finish')
    pdf_queque.join()
    print('{0} files {1}'.format(num_pdfs, proc_desc))
    if callback_func is not None:
        callback_func(*tuple(args))


def multi_thread_do(q, func, *args):
    while True:
        p = q.get()
        try:
            func(p, *args)
        except:
            print u'error doing {0} on {1}'.format(func, p)
        q.task_done()


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
                'src': fn,
                'sid': ann['sid'],
                'text': ann['text'],
                'struct': '' if 'struct' not in ann else ann['struct'],
                'sapienta': '' if 'CoreSc' not in ann else ann['CoreSc'],
                'entities': '' if 'ncbo' not in ann else ' '.join( list(set([a['annotation']['text'].lower() for a in ann['ncbo']])) )
              }
        if 'marked' in ann:
            co['marked'] = ann['marked']
            hts.append(co)
        else:
            co['marked'] = ''
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


def load_json_data(file_path):
    data = None
    with codecs.open(file_path, encoding='utf-8') as rf:
        data = json.load(rf, encoding='utf-8')
    return data

def save_sentences(non_hts, hts, output_path):
    training_testing_ratio = 0.6
    total_num = min(len(hts), len(non_hts))
    trainin_len = int(training_testing_ratio * total_num)
    save_json_array(non_hts, join(output_path, 'full_non_hts.json'))
    save_json_array(hts, join(output_path, 'full_hts.json'))

    save_json_array(non_hts[:trainin_len], join(output_path, 'non_hts.json'))
    save_json_array(hts[:trainin_len], join(output_path, 'hts.json'))
    save_json_array(non_hts[trainin_len:total_num], join(output_path + "/test", 'non_hts.json'))
    save_json_array(hts[trainin_len:total_num], join(output_path + "/test", 'hts.json'))

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


def main():
    ann_to_training('./anns_v2', './training')
    # sents = [
    #     'The control group was comprised of 15 elderly community dwelling individuals of comparable age and educational background',
    #     'This resulted in data of 172 participants to be included in the present study.'
    # ]
    # relation_patterns(sents[0])

if __name__ == "__main__":
    main()
