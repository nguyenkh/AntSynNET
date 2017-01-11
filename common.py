from itertools import count
from collections import defaultdict, OrderedDict
import random
import cPickle
import numpy
from numpy import fromstring, dtype
import theano

class Resources():
    """
    Holds the resources
    """
    def __init__(self, resource_prefix):
        """
        Initialize the resources
        """
        self.term_to_id = cPickle.load(open(resource_prefix + '_term_to_id.p', 'rb'))
        self.id_to_term = cPickle.load(open(resource_prefix + '_id_to_term.p', 'rb'))
        self.path_to_id = cPickle.load(open(resource_prefix + '_path_to_id.p', 'rb'))
        self.id_to_path = cPickle.load(open(resource_prefix + '_id_to_path.p', 'rb'))
        self.pattern = cPickle.load(open(resource_prefix + '_patterns.p', 'rb'))

    def get_term_by_id(self, id):
        return self.id_to_term[str(id)]

    def get_path_by_id(self, id):
        return self.id_to_path[str(id)]

    def get_id_by_term(self, term):
        return int(self.term_to_id[term]) if self.term_to_id.has_key(term) else -1

    def get_id_by_path(self, path):
        return int(self.path_to_id[path]) if self.path_to_id.has_key(path) else -1

    def get_patterns(self, x, y):
        """
        Returns the relations from x to y
        """
        pattern_dict = {}
        key = str(x) + '###' + str(y)
        path_str = self.pattern[key] if self.pattern.has_key(key) else ''

        if len(path_str) > 0:
            paths = [tuple(map(int, p.split(':'))) for p in path_str.split(',')]
            pattern_dict = { path : count for (path, count) in paths }

        return pattern_dict

def preprocess_instance(instance, maxlen=12):
    paths = instance.keys()
    n_samples = len(paths)

    l = numpy.zeros((maxlen, n_samples)).astype('int64')
    p = numpy.zeros((maxlen, n_samples)).astype('int64')
    g = numpy.zeros((maxlen, n_samples)).astype('int64')
    d = numpy.zeros((maxlen, n_samples)).astype('int64')
    mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    c = numpy.zeros((n_samples, n_samples)).astype('int64')

    for idx, path in enumerate(paths):
        c[idx,idx] = instance[path]
        for i,node in enumerate(path):
            l[i,idx] = node[0]
            p[i,idx] = node[1]
            g[i,idx] = node[2]
            d[i,idx] = node[3]
            mask[i,idx] = 1.

    return l, p, g, d, mask, c   

def load_data(corpus_prefix, dataset_prefix, embeddings_file=None):
    '''
    Loads the resources
    '''
    print "Loading dataset..."
    train_set = load_dataset(dataset_prefix + '.train')
    test_set = load_dataset(dataset_prefix + '.test')
    valid_set = load_dataset(dataset_prefix + '.val')
    y_train = [int(train_set[key]) for key in train_set.keys()]
    y_test = [int(test_set[key]) for key in test_set.keys()]
    y_valid = [int(valid_set[key]) for key in valid_set.keys()]
    
    dataset_keys = train_set.keys() + test_set.keys() + valid_set.keys()
    
    vocab = OrderedDict()
    for kk in dataset_keys:
        vocab[kk[0]]
        vocab[kk[1]]
        
    print 'Initializing word embeddings...' 
      
    if embeddings_file is not None:
        vecs, words = load_embeddings(embeddings_file, vocab)    
        key_pairs, patterns, lemma, pos, dep, dist = load_patterns(corpus_prefix, dataset_keys, words)     
    else:
        key_pairs, patterns, lemma, pos, dep, dist = load_patterns(corpus_prefix, dataset_keys) 
        vecs = None
    
    print 'Number of lemmas: %d, POS tags: %d, dependency labels: %d, distance labels: %d'\
          %(len(lemma), len(pos), len(dep), len(dist))

    X_train = patterns[:len(train_set)]
    X_test = patterns[len(train_set):len(train_set)+len(test_set)]
    X_valid = patterns[len(train_set)+len(test_set):]
    
    train_key_pairs = key_pairs[:len(train_set)]
    test_key_pairs = key_pairs[len(train_set):len(train_set)+len(test_set)]
    valid_key_pairs = key_pairs[len(train_set)+len(test_set):]

    train = (X_train, y_train)
    test = (X_test, y_test)
    valid = (X_valid, y_valid)
    keys = (train_key_pairs, test_key_pairs, valid_key_pairs)
        
    return vecs, train, valid, test, keys, lemma, pos, dep, dist

def load_patterns(corpus_prefix, dataset_keys, words=None):
    """
    Loads patterns from the database
    """

    # Define the dictionaries
    if words is None:     
        lemma = defaultdict(count(0).next)
        pre_trained_embs = False
        lemma['#UNKNOWN#']
    else:
        lemma = words
        pre_trained_embs = True
        
    pos = defaultdict(count(0).next)
    dep = defaultdict(count(0).next)
    dist = defaultdict(count(0).next)
    
    pos['#UNKNOWN#']
    dep['#UNKNOWN#']
    dist['#UNKNOWN#']

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = Resources(corpus_prefix)

    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    patterns_x_to_y = [{ vectorize_path(path, lemma, pos, dep, dist, pre_trained_embs) : count
                      for path, count in get_patterns(corpus, x_id, y_id).iteritems() }
                    for (x_id, y_id) in keys]
    patterns_x_to_y = [ { p : c for p, c in patterns_x_to_y[i].iteritems() if p is not None } for i in range(len(keys)) ]

    patterns = patterns_x_to_y

    empty = [dataset_keys[i] for i, path_list in enumerate(patterns) if len(path_list.keys()) == 0]
    print 'Pairs without patterns:', len(empty), ', Dataset size:', len(dataset_keys)

    # Get the index for x and y (get a lemma index)
    if words is None:
        key_pairs = [(lemma[x], lemma[y]) for (x, y) in dataset_keys]
    else:
        key_pairs = [(lemma.get(x, 0), lemma.get(y, 0)) for (x, y) in dataset_keys]

    return key_pairs, patterns, lemma, pos, dep, dist

def get_patterns(corpus, x, y):
    """
    Get the paths between x and y
    """
    x_to_y_patterns = corpus.get_patterns(x, y)
    patterns = { corpus.get_path_by_id(path) : count for (path, count) in x_to_y_patterns.iteritems() }
    
    return patterns


def vectorize_path(path, lemma, pos, dep, dist, pre_trained_embs):
    """
    Returns a pattern
    """
    vectorized_path = [vectorize_node(node, lemma, pos, dep, dist, pre_trained_embs) for node in path.split(':::')]
    if None in vectorized_path:
        return None
    else:
        return tuple(vectorized_path)

def vectorize_node(node, lemma, pos, dep, dist, pre_trained_embs):
    """
    Returns a node: concatenate lemma/pos/dep/dist
    """
    try:
        l, p, g, d = node.split('/')
    except:
        return None
    
    if pre_trained_embs:
        return tuple([lemma.get(l, 0), pos[p], dep[g], dist[d]])
    else:
        return tuple([lemma[l], pos[p], dep[g], dist[d]])

def load_dataset(dataset_file):
    """
    Loads the dataset
    """
    with open(dataset_file, 'r') as fin:
        lines = [tuple(line.strip().split('\t')) for line in fin]
        random.shuffle(lines)
        dataset = { (x, y) : label for (x, y, label) in lines }

    return dataset

def load_embeddings(file_name, vocab):
    """
    Load the pre-trained embeddings from a file
    """
    words = []
    embs = []
    
    with open(file_name, 'rb') as f:
        if file_name.endswith('.txt'):
            header = to_unicode(f.readline())
            if len(header.split()) == 2: vocab_size, vector_size = map(int, header.split())
            elif len(header.split()) > 2:
                parts = header.rstrip().split(" ")
                word, vec = parts[0], list(map(numpy.float32, parts[1:]))
                words.append(to_unicode(word))
                embs.append(vec)
            for _, line in enumerate(f):
                parts = to_unicode(line.rstrip()).split(" ")
                word, vec = parts[0], list(map(numpy.float32, parts[1:]))
                words.append(to_unicode(word))
                embs.append(vec)
        elif file_name.endswith('.bin'):
            header = to_unicode(f.readline())
            vocab_size, vector_size = map(int, header.split())
            binary_len = dtype(numpy.float32).itemsize * vector_size
            for _ in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':
                        word.append(ch)
                word = to_unicode(b''.join(word))
                words.append(word)
                vec = fromstring(f.read(binary_len), dtype=numpy.float32)
                embs.append(vec)
        else:
            print "The extension of embeddings file is either .bin or .txt"
    
    # Add the unknown word
    embs_dim = len(embs[1])
    
    for word in vocab.keys():
        if word not in words:
            words.append(word)
            unk_vec = numpy.random.uniform(-0.25,0.25,embs_dim)
            embs = numpy.vstack(embs, unk_vec)
             
    UNKNOWN_WORD = numpy.random.uniform(-0.25,0.25,embs_dim)
    wv = numpy.vstack((UNKNOWN_WORD, embs))
    words = ['#UNKNOWN#'] + list(words)
    
    word_index = { w : i for i, w in enumerate(words) }

    return wv, word_index

def to_unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    else:
        return unicode(text, encoding=encoding, errors=errors)
    