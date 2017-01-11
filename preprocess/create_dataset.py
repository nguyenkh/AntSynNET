import random
import numpy
import math
import argparse
from common import Resources

NEG_POS_RATIO = 1
TRAIN_PORTION = 0.7
TEST_PORTION = 0.25
VALID_PORTION = 0.05

def main():
    """
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus', type=str)
    parser.add_argument('-pos', type=str)
    parser.add_argument('-neg', type=str)
    parser.add_argument('-prefix', type=str)
    parser.add_argument('-min', type=int)
    
    args = parser.parse_args()
    corpus_prefix = args.corpus
    pos_file = args.pos
    neg_file = args.neg
    dataset_prefix = args.prefix
    min_occurrences = args.min
    
    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = Resources(corpus_prefix)
    
    print 'Loading the dataset...'
    pos_pairs = load_dataset(pos_file)
    neg_pairs = load_dataset(neg_file)
    
    print 'Filtering out word-pairs...'
    filtered_pos_pairs = filter_word_pairs(corpus, pos_pairs.keys(), min_occurrences)
    filtered_neg_pairs = filter_word_pairs(corpus, neg_pairs.keys(), min_occurrences)
    
    positives, negatives = keep_ratio(filtered_pos_pairs, filtered_neg_pairs)
    
    pos_train_set, pos_test_set, pos_valid_set = split_dataset(positives, pos_pairs)
    neg_train_set, neg_test_set, neg_valid_set = split_dataset(negatives, neg_pairs)
    
    train_set_x = pos_train_set[0] + neg_train_set[0]
    train_set_y = pos_train_set[1] + neg_train_set[1]
    test_set_x = pos_test_set[0] + neg_test_set[0]
    test_set_y = pos_test_set[1] + neg_test_set[1]
    valid_set_x = pos_valid_set[0] + neg_valid_set[0]
    valid_set_y = pos_valid_set[1] + neg_valid_set[1]
        
    with open(dataset_prefix + '.train', 'wb') as f:
        for (x,y), label in zip(train_set_x, train_set_y):
            st = x + '\t' + y + '\t' + str(label) + '\n'
            f.write(st)
    with open(dataset_prefix + '.test', 'wb') as f:
        for (x,y), label in zip(test_set_x, test_set_y):
            st = x + '\t' + y + '\t' + str(label) + '\n'
            f.write(st)
    with open(dataset_prefix + '.valid', 'wb') as f:
        for (x,y), label in zip(valid_set_x, valid_set_y):
            st = x + '\t' + y + '\t' + str(label) + '\n'
            f.write(st)
            
    print 'Done........!'
    
def keep_ratio(pos_pairs, neg_pairs):
    """
    Keeps the ratio between postive and negative
    """
    if len(neg_pairs) > len(pos_pairs) * NEG_POS_RATIO:
        negatives = random.sample(neg_pairs, len(neg_pairs) * NEG_POS_RATIO)
        random.shuffle(pos_pairs)
        positives = pos_pairs
    else:
        positives = random.sample(pos_pairs, int(math.ceil(len(neg_pairs) / NEG_POS_RATIO)))
        random.shuffle(neg_pairs)
        negatives = neg_pairs
        
    return positives, negatives

def load_dataset(infile):
    """
    Loads dataset file
    """
    with open(infile, 'rb') as f:
        lines = [tuple(line.strip().lower().split('\t')) for line in f]
        dataset = {(x,y) : int(label) for (x,y,label) in lines}
    return dataset

def filter_word_pairs(corpus, dataset_keys, min_occurrences=5):
    """
    Filter out pairs from the dataset, to keep only those with enough path occurrences in the corpus
    """
    first_filter = [(x, y) for (x, y) in dataset_keys if len(x) > 2 and len(y) > 2 and
                    len(set(x.split(' ')).intersection(y.split(' '))) == 0]
    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in first_filter]
    paths_x_to_y = [set(get_patterns(corpus, x_id, y_id)) for (x_id, y_id) in keys]
    filtered_keys = [first_filter[i] for i, key in enumerate(keys) if len(paths_x_to_y[i]) >= min_occurrences]

    return filtered_keys

def get_patterns(corpus, x, y):
    """
    Returns the patterns between (x, y) term-pair
    """
    x_to_y_paths = corpus.get_relations(x, y)
    paths = [path_id for path_id in x_to_y_paths.keys()]
    return paths

def split_dataset(word_pairs, dict_word_pairs):
    """
    Splits dataset to train, test and valid sets
    """
    data_x = word_pairs
    data_y = [dict_word_pairs[(x,y)] for (x,y) in data_x]
    
    n_samples = len(data_x)
    n_train = int(numpy.round(n_samples * TRAIN_PORTION))
    n_valid = int(numpy.round(n_samples * (1. - VALID_PORTION)))
    
    sidx = numpy.random.permutation(n_samples)
    train_set_x = [data_x[s] for s in sidx[ : n_train]]
    train_set_y = [data_y[s] for s in sidx[ : n_train]]
    test_set_x = [data_x[s] for s in sidx[n_train : n_valid]]
    test_set_y = [data_y[s] for s in sidx[n_train : n_valid]]
    valid_set_x = [data_x[s] for s in sidx[n_valid : ]]
    valid_set_y = [data_y[s] for s in sidx[n_valid : ]]
    
    train_set = (train_set_x, train_set_y)
    test_set = (test_set_x, test_set_y)
    valid_set = (valid_set_x, valid_set_y)
    
    return train_set, test_set, valid_set

if __name__=='__main__':
    main()
    