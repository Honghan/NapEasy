import ann_utils as utils
from os.path import join, isfile
from os import listdir
from time import time
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def load_ht_data(ann_file_path):
    score_files = [join(ann_file_path, f) for f in listdir(ann_file_path) if isfile(join(ann_file_path, f))
                   and f.endswith('_annotated_ann.json')]
    sents = []
    for sf in score_files:
        sents += [{'text': so['text'], 'class': 'ht' if 'marked' in so else 'nht'}
                  for so in (sos for sos in utils.load_json_data(sf))]
    return sents
    print 'total #sents %s \n top 1 is %s' % (len(sents), sents[0])


def vect_data(train_file_path, test_file_path):
    sents = load_ht_data(train_file_path)
    do_balanced_training = False
    if not do_balanced_training:
        train_text_list = [sent['text'] for sent in sents]
        train_labels = [sent['class'] for sent in sents]
    else:
        train_text_list = []
        train_labels = []
        count_ht = 0
        count_nht = 0
        for sent in sents:
            if sent['class'] == 'ht':
                count_ht += 1
                train_text_list.append(sent['text'])
                train_labels.append(sent['class'])
        for sent in sents:
            if sent['class'] == 'nht':
                train_text_list.append(sent['text'])
                train_labels.append(sent['class'])
                count_nht += 1
                if count_nht >= count_ht:
                    break

    test_sents = load_ht_data(test_file_path)
    test_text_list = [sent['text'] for sent in test_sents]
    test_labels = [sent['class'] for sent in test_sents]

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    train_data = vectorizer.fit_transform(train_text_list)
    test_data = vectorizer.transform(test_text_list)
    print("train data [n_samples: %d, n_features: %d]" % train_data.shape)
    print("test data [n_samples: %d, n_features: %d]" % test_data.shape)
    return train_data, train_labels, test_data, test_labels


def benchmark(clf, train_data, train_labels, test_data, test_labels):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(train_data, train_labels)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test_data)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    f1_score = metrics.f1_score(test_labels, pred, pos_label='ht')
    score = metrics.accuracy_score(test_labels, pred)
    precision = metrics.precision_score(test_labels, pred, pos_label='ht')
    recall = metrics.recall_score(test_labels, pred, pos_label='ht')
    print("accuracy/F1/Precision/Recall:   %0.3f\t%0.3f\t%0.3f\t%0.3f" % (score, f1_score, precision, recall))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


def main():
    train_data_path = './local_exp/anns_v2/'
    test_data_path = './local_exp/20-test-papers'
    print 'loading data...'
    (train_data, train_labels, test_data, test_labels) = vect_data(train_data_path, test_data_path)
    print 'data loaded. learning...'
    results = []
    for clf, name in (
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, train_data, train_labels, test_data, test_labels))


if __name__ == "__main__":
    main()
