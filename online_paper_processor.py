import requests
import xml.etree.ElementTree as et
import nltk.data
import ann_utils as utils
from os.path import join, isdir, exists
from os import makedirs, getpid, listdir
import auto_highlighter as ah
import datetime
import json
import smtplib
from multiprocessing import Process
import time
import traceback


europepmc_full_text_url = 'http://www.ebi.ac.uk/europepmc/webservices/rest/{}/fullTextXML'
napeasy_api_url = 'http://napeasy.org/napeasy_api/api'
napeasy_key = ''
working_path = './local_exp/jobs/'
num_processes = 5
no_semantic_fix = False

notification_email_template = """Dear {user},

You have submitted a highligh job at napeasy.org. Now, your highlight job (id: {jobid}) has finished.
The result can be viewed at http://napeasy.org/ht.html?{jobid}.

If you have any questions, please contact honghan.wu@gmail.com.

Yours sincerely,
NapEasy team
http://napeasy.org
"""

job_init_email_template = """Dear {user},

Napeasy.org is processing your highlight job (id: {jobid}). The progress will be updated at http://napeasy.org/ht.html?{jobid}.

You will receive a notification email when it's done. If you have any questions, please contact honghan.wu@gmail.com.

Yours sincerely,
NapEasy team
http://napeasy.org
"""


class status_code:
    DONE = 200
    INITIATED = 101
    SCORING = 102 # scoring sentences
    SEMANTIC_FIXING = 103 # semantic fixing
    HIGHLIGHTING = 104 # highlighting sentences

    ERROR_FS = 501 # file system error
    ERROR_HT_PREPROCESSING= 502 # highlight preprocessing error
    ERROR_PMCID_NOT_FOUND = 503 # full text not avaialble


def sent_email_notification(email, title, msg_body):
    fromaddr = 'napeasy.noreply@gmail.com'
    toaddrs  = email
    msg = "\r\n".join([
        "From: " + fromaddr,
        "To: " + email,
        "Subject: %s" % title,
        "",
        msg_body
    ])
    username = 'napeasy.noreply@gmail.com'
    password = '123321!a'
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()


def post_data(url, postobj, authobj=None):
    response = ''
    if authobj is None:
        r = requests.post(url, data=postobj)
        response = r.content
    else:
        r = requests.post(url, data=postobj, auth=authobj)
        response = r.content
    return response


def get_pmc_paper_fulltext(pmcid):
    s = ''
    full_xml = requests.get(europepmc_full_text_url.format(pmcid)).content
    try:
        root = et.fromstring(full_xml)
        # print iter_element_text(root)
        telem = root.find('.//article-title')
        title = ''
        if telem is not None:
            title = ''.join(telem.itertext())
            title += '.\n'
        elem = root.find('body')
        s = title + iterate_get_text(elem)
    except Exception, e:
        traceback.print_exc()
    return s


def iterate_get_text(elem):
    """
    get all inner text values of this element with special cares of
    1) ignoring not relevant nodes, e.g., xref, table
    2) making section/title texts identifiable by the sentence tokenizer
    Args:
        elem:

    Returns:

    """
    remove_tags = ['table']
    line_break_tags = ['sec', 'title']
    s = ''
    if elem.tag not in remove_tags:
        s += elem.text.strip() + ' ' if elem.text is not None else ''
        if elem.tag in line_break_tags and elem.text is not None:
            s += '.\n'
        for e in list(elem):
            ct = iterate_get_text(e)
            s += (' ' + ct) if len(ct) > 0 else ''
    s += elem.tail.strip() + ' ' if elem.tail is not None else ''
    return s


def process_pmc_paper(pmcid, job_path, job_id):
    ann_file = join(job_path, pmcid + '_ann.json')
    if exists(ann_file):
        print '%s exists, skipping download' % ann_file
        update_paper_fulltext(pmcid, utils.load_json_data(ann_file))
        return
    t = get_pmc_paper_fulltext(pmcid)
    if t is None or len(t) == 0:
        utils.append_text_file(pmcid, join(job_path, 'not_available.txt'))
    else:
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_detector.tokenize(t.strip())
        if job_path is not None:
            fulltext = [{'text': sents[i], 'sid': str(i+1)} for i in range(len(sents))]
            utils.save_json_array(fulltext,
                                  ann_file)
            update_paper_fulltext(pmcid, fulltext)


def get_email_addr(job_path):
    if exists(join(job_path, 'email.txt')):
        lines = utils.load_text_file(join(job_path, 'email.txt'))
        return lines[0] if lines[0] != 'napeasy.noreply@gmail.com' else None
    else:
        return None


def update_job_progress(job_path, jobid, s_code, message, result=None):
    report = {'jobid': jobid, 'status': s_code, 'message': message, 'time': str(datetime.datetime.now()),
              'result': result}
    # save status locally
    utils.append_text_file(json.dumps(report) + '\n', join(job_path, 'status.json'))
    print post_data(napeasy_api_url, {'r': 'updateJobStatus', 'key': napeasy_key, 'data': json.dumps(report)})
    # sent an email when the job started
    if s_code == status_code.INITIATED:
        user_email = get_email_addr(job_path)
        if user_email is not None:
            sent_email_notification(user_email,
                                    '[napeasy update] Your highlight job started at napeasy.org',
                                    job_init_email_template.format(**{"user": user_email,
                                                                          "jobid": jobid}))
    # notification email when the job finishes.
    if s_code in \
            [status_code.DONE, status_code.ERROR_FS, status_code.ERROR_HT_PREPROCESSING, status_code.ERROR_PMCID_NOT_FOUND] \
            and exists(join(job_path, 'email.txt')):
        user_email = get_email_addr(job_path)
        if user_email is not None:
            sent_email_notification(user_email,
                                    '[napeasy update] Your highlight job finished',
                                    notification_email_template.format(**{"user": user_email,
                                                                          "jobid": jobid}))
            print 'email notificaiton sent to %s' % user_email


def update_paper_fulltext(pmcid, fulltext):
    print post_data(napeasy_api_url, {'r': 'updatePaperFulltext', 'pmcid': pmcid, 'key': napeasy_key, 'fullText': json.dumps(fulltext)})


def update_paper_summ(pmcid, summ):
    print post_data(napeasy_api_url, {'r': 'updatePaperSumm', 'pmcid': pmcid, 'key': napeasy_key, 'summ': json.dumps(summ)})


def update_score_path_summ(score_path):
    # regenerate sum because of new score file after semantic fixing
    hter = ah.HighLighter.get_instance()
    sum_files = utils.filter_path_file(score_path, 'sum')
    for s in sum_files:
        pmcid = s[:s.rfind('_')]
        print join(score_path[:score_path.rfind('/summ')], pmcid + '_ann.json')
        ah.summ(hter, join(score_path[:score_path.rfind('/summ')], pmcid + '_ann.json'), score_path)
        update_paper_summ(pmcid, utils.load_json_data(join(score_path, s)))
        print 'paper %s summary uploaded' % pmcid


def iterate_job_to_update_summ(job_path):
    sum_folders = [f + '/summ' for f in listdir(job_path)
                   if isdir(join(job_path, f))
                   and isdir(join(job_path, f + '/summ'))]
    for sf in sum_folders:
        folder = join(job_path, sf)
        update_score_path_summ(folder)
        print '%s done' % folder


def get_job_path_id_from_score_path(score_path):
    job_path = score_path[:score_path.rfind('/')]
    job_id = job_path[job_path.rfind('/') + 1:]
    return job_path, job_id


def finish_highlighting(container, score_path, hter, threshold, manual_ann):
    print container
    job_path, job_id = get_job_path_id_from_score_path(score_path)
    update_job_progress(job_path, job_id, status_code.DONE, 'Job Done', container)


def do_summary_job(job_path, jobid):
    score_path = join(job_path, 'summ')
    if not exists(score_path):
        makedirs(score_path)
    update_job_progress(job_path, jobid, status_code.SCORING, 'scoring sentences...')
    # ah.summarise_all_papers(job_path, score_path, callback=(do_highlighting if no_semantic_fix else do_semantic_fixing))
    ah.multiple_processing_summarise_papers(job_path,
                                            score_path,
                                            callback=(do_highlighting if no_semantic_fix
                                                      else do_semantic_fixing))


def do_semantic_fixing(score_path):
    job_path, job_id = get_job_path_id_from_score_path(score_path)
    update_job_progress(job_path, job_id, status_code.SEMANTIC_FIXING, 'semantically matching language patterns...')
    # utils.semantic_fix_all_scores(score_path, cb=do_highlighting_after_fixing)
    utils.multi_processing_semantic_fix_all_scores(score_path, cb=lambda score_dir: do_highlighting(score_dir))


def do_highlighting_after_fixing(sp_patterns, sp_cats, hter, score_path):
    do_highlighting(score_path)


def do_highlighting(score_path):
    job_path, job_id = get_job_path_id_from_score_path(score_path)
    update_score_path_summ(score_path)
    update_job_progress(job_path, job_id, status_code.HIGHLIGHTING, 'highlighting...')
    threshold = .4
    ret_container = []
    hter = ah.HighLighter.get_instance()
    utils.multi_thread_process_files(score_path, '', 3, ah.score_paper_threshold,
                                     args=[ret_container, score_path, hter, threshold, None],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=finish_highlighting)


def do_user_job(job_id, working_path, pmcids, user_email=None):
    job_path = join(working_path, job_id)
    # create job folder if not exists
    if not isdir(job_path):
        if exists(job_path):
            update_job_progress(job_path, job_id, status_code.ERROR_FS, 'job path exists but it''s not a directory')
        else:
            makedirs(job_path)
    if user_email is not None:
        utils.save_text_file(user_email, join(job_path, 'email.txt'))
    if len(pmcids) == 0 or (len(pmcids) == 1 and len(pmcids[0]) == 0):
        update_job_progress(job_path, job_id, status_code.DONE, 'Job Done')
    else:
        update_job_progress(job_path, job_id, status_code.INITIATED, 'job started, downloading papers...')
        utils.multi_thread_tasking(pmcids, min(5, len(pmcids)), process_pmc_paper,
                                   args=[job_path, job_id],
                                   callback_func=do_summary_job)


def do_job():
    idle_times = 0
    while True:
        try:
            s = requests.get(napeasy_api_url + '?key=' + napeasy_key + '&r=retrieveAJob').content
            ret_data = json.loads(s)
            if ret_data['data'] is not None:
                idle_times = 0
                job = json.loads(ret_data['data'])
                print job
                do_user_job(job['jobid'], working_path, job['pmcids'].split(','), user_email=job['email'])
            else:
                idle_times += 1
                if idle_times % 10 == 0:
                    #print '..%s..' % getpid()
                    idle_times = 0
        except:
            print 'retrieving/working on job failed'
            traceback.print_exc()
        time.sleep(5)


def multi_processing_job():
    processes = []
    for i in range(num_processes):
        p = Process(target=do_job)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print 'all done'


if __name__ == "__main__":
    # working_path = './local_exp/jobs/'
    # do_user_job('j001', working_path, ['PMC5116532'])
    # process_pmc_paper('PMC5116532', './local_exp/jobs/j001')
    # multi_processing_job()
    iterate_job_to_update_summ('')
