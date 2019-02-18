import codecs
import numpy as np

from docopt import docopt
from sklearn.metrics import average_precision_score
import sys

def main():
    """
    Calculate the Average Precision (AP) at k.
    """

    # Get the arguments
    file_list = sys.argv[1:]
    for f in file_list:

        test_results_file = f
        cutoff = 0

        # Sort the lines in the file in descending order according to the score
        dataset = load_dataset(test_results_file)
        dataset = sorted(dataset, key=lambda line: line[-1], reverse=True)

        gold = np.array([1 if label == 'True' else 0 for (x, y, label, score) in dataset])
        scores = np.array([score for (x, y, label, score) in dataset])

        for i in range(1, min(cutoff + 1, len(dataset))):
            try:
                score = average_precision_score(gold[:i], scores[:i])
            except:
                score = 0
            # print 'Average Precision at %d is %.3f' % (i, 0 if score == -1 else score)

        print (f, '  FINAL: Average Precision at %d is %.3f' % (len(dataset), average_precision_score(gold, scores)))


def load_dataset(dataset_file):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = [tuple(line.strip().split('\t')) for line in f_in]
        if len(dataset[0]) == 5:
            dataset = [(x, y, label, score) for (x, y, label, relation, score) in dataset]
        dataset = [(x, y, label, float(score)) for (x, y, label, score) in dataset]

    return dataset


if __name__ == "__main__":
    main()
