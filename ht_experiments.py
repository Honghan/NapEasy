import auto_highlighter as ah
import ann_utils as utils
import numpy as np
import sys


# folders of paper groups
folder_200_papers = './summaries/'
folder_18_papers = './20-test-papers/summaries/'
folder_10_manual_checked = './10-manual-checked/summaries/'

# manual checked result file
manual_file = './results/manual_annotations.json'


def dump_file_results(files, out_file):
    ht = ah.HighLighter.get_instance()
    ctn = []
    s = 'sid\thighlighted\tpredicted\ttype\toverall score\tsub-pred score/confidence\tCD Score\tNE Score\ttext\n'
    for f in files:
        s += '\n\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
            '**','**','**','**','**','**','**','**', f)
        s += '\n'.join(ah.score_paper_threshold(f, ctn, '', ht, 0.07))
        print '%s done' % f
    utils.save_text_file(s, out_file)


def output_random_picked_files():
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
    dump_file_results(test_files, './results/sample_test_files_full.tsv')


def pp_score_exp(container, out_file, hter, threshold, manual_ann):
    should = 0
    correct = 0
    predicted = 0
    total = 0

    print 'precision\trecall\t#highlighted\t#predicted\tpaper'
    for p in container:
        should += p['hts']
        correct += p['correct']
        predicted += p['predicted']
        total += p['max_sid'] if 'max_sid' in p else 0
        print '{:.2f}\t{:.2f}\t{}\t{}\t{}'.format(1.0 * p['correct'] / p['predicted'] if p['predicted'] > 0 else 0,
                                      1.0 * p['correct'] / p['hts'],
                                      p['hts'], p['predicted'], p['paper'])

    if predicted == 0:
        print '{}\t-\t-\t-'.format(threshold)
    else:
        precision = 1.0 * correct / predicted
        recall = 1.0 * correct / should
        print '{}\t{}\t{}\t{}\t{}'.format(threshold, precision, recall,
                                          '-' if total == 0 else (1.0 * predicted - correct)/(total - correct),
                                          2 * precision * recall / (precision + recall))
        # utils.save_json_array(container, out_file)


def score_exp(score_files_path, out_file, threshold, manual_ann=None):
    ret_container = []
    hter = ah.HighLighter.get_instance()
    utils.multi_thread_process_files(score_files_path, '', 1, ah.score_paper_threshold,
                                     args=[ret_container, out_file, hter, threshold, manual_ann],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_score_exp)


def get_manual_checked_result():
    return utils.load_json_data(manual_file)


def exp_given_threshold(corpus_path, threshold, manual_ann=None):
        score_exp(corpus_path, '', threshold, manual_ann)


def exp_iterating_threshold(corpus_path, manual_ann=None):
    print 'precision\trecall\tfall out\t#f measure'
    for i in np.arange(0.1, 1.1, 0.100):
        score_exp(corpus_path, '', i, manual_ann)


# highlighting post-processing - saving results
def pp_highlight(container, out_file, hter, threshold, manual_ann):
    utils.save_json_array(container, out_file)


# do highlights
def do_highlighting(score_path):
    threshold = .4
    ret_container = []
    hter = ah.HighLighter.get_instance()
    utils.multi_thread_process_files(score_path, '', 3, ah.score_paper_threshold,
                                     args=[ret_container, score_path + '/highlight-results.json', hter, threshold, None],
                                     file_filter_func=lambda fn: fn.endswith('_scores.json'),
                                     callback_func=pp_highlight)


# highlight papers in a given folder
def highlight_papers(ann_path, score_path):
    ah.summarise_all_papers(ann_path, score_path, callback=do_highlighting)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == 'ht':
        highlight_papers(sys.argv[2], sys.argv[3])
    else:
        # exp_iterating_threshold(folder_18_papers)
        # exp_iterating_threshold(folder_10_manual_checked, get_manual_checked_result())
        # exp_iterating_threshold(folder_200_papers)
        exp_given_threshold(folder_200_papers, .4)
        exp_given_threshold(folder_18_papers, .4)
        exp_given_threshold(folder_10_manual_checked, .4, get_manual_checked_result())
