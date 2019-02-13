# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)

import numpy as np
import pdb
import nltk


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if (string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    return lengths[len(string)][len(sub)]


class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set

    '''

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        if len(candidate) * len(refs) == 0:
            return 0.0

        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        # token_c = candidate[0].split(" ")
        token_c = nltk.word_tokenize(candidate[0].strip().lower())
        if len(token_c) == 0:
            return 0.0

        for reference in refs:
            # split into tokens
            # token_r = reference.split(" ")
            token_r = nltk.word_tokenize(reference.strip().lower())
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)

            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))
        prec_max = max(prec)
        rec_max = max(rec)

        if (prec_max != 0 and rec_max != 0):
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def method(self):
        return "Rouge"


# Rouge().calc_score(candidate, refs)
# Rouge().calc_score(["a b"], ["a b c"])



"""
import json
from myRouge import Rouge

path_to_reference_file = "sample_test_data/dev_as_references.json"
path_to_candidate_file = "sample_test_data/dev_first_sentence_as_candidates.json"

reference_list = []
with open(path_to_reference_file, 'r', encoding='utf-8') as data_file:
    for line in data_file:
        json_object = json.loads(line)
        reference_list.append(json_object["answers"])

candidate_list = []
with open(path_to_candidate_file, 'r', encoding='utf-8') as data_file:
    for line in data_file:
        json_object = json.loads(line)
        candidate_list.append(json_object["answers"])

all_scores = []
for i, _ in enumerate(reference_list):
    if len(reference_list[i]) == 0:
        continue
    all_scores.append(Rouge().calc_score(candidate_list[i], reference_list[i]))

sum(all_scores)/len(all_scores)
# 0.1201400967262593

# 【参考】rouge_l: 0.12094912964570531
"""