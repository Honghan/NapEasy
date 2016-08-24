import json
import codecs
import article_ann as pann
import matplotlib.pyplot as plt
import random
import os


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

if __name__ == "__main__":
    af = './anns/t_ann.json'
    analysis_ann(af)
