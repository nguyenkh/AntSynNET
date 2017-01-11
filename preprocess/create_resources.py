import gzip
import cPickle
import argparse
from collections import defaultdict
from itertools import count

def main():
    """
    Creates the resource from triplets file
    Usage:
        create_resource.py -input <triplets_file> -freq <frequent_paths_file> -prefix <resource_prefix>

        <triplets_file> = the file that contains text triplets, formated as X\tY\tpath
        <frequent_paths_file> = the file containing the frequent paths. It could be computed using 
        the triplet files created from parse_wikipedia.py (e.g. parsed_corpus):
        sort -u parsed_corpus | cut -f3 -d$'\t' > paths
        awk -F$'\t' '{a[$1]++; if (a[$1] == 5) print $1}' paths > frequent_paths
        <resource_prefix> = the file names' prefix for the resource files
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str)
    parser.add_argument('-freq', type=str)
    parser.add_argument('-prefix', type=str)
    args = parser.parse_args()
    
    triplets_file = args.input
    frequent_paths_file = args.freq
    resource_prefix = args.prefix
    
    # Load the frequent paths
    with gzip.open(frequent_paths_file, 'rb') as f:
        frequent_paths = set([line.strip() for line in f])
    print 'The number of frequent paths: %d' %len(frequent_paths)
    
    paths = list(set(frequent_paths))
    left = defaultdict(count(0).next)
    right = defaultdict(count(0).next)
    # Load the corpus
    with gzip.open(triplets_file, 'rb') as f:
        for line in f:
            if len(line.strip().split('\t'))==3:
                l, r, p = line.strip().split('\t')
                left[l]
                right[r]
    print 'Read triples successfully!'
          
    entities = list(set(left.keys()).union(set(right.keys())))
    term_to_id = { t : i for i, t in enumerate(entities) }
    path_to_id = { p : i for i, p in enumerate(paths) }

    # Terms
    term_to_id_db = {}
    id_to_term_db = {}
    
    for term, id in term_to_id.iteritems():
        id, term = str(id), str(term)
        term_to_id_db[term] = id
        id_to_term_db[id] = term
    
    cPickle.dump(term_to_id_db, open(resource_prefix + '_term_to_id.p', 'wb'))
    cPickle.dump(id_to_term_db, open(resource_prefix + '_id_to_term.p', 'wb'))       
    print 'Created term databases...'

    # Paths
    path_to_id_db = {}
    id_to_path_db = {}
    
    for path, id in path_to_id.iteritems():
        id, path = str(id), str(path)
        path_to_id_db[path] = id
        id_to_path_db[id] = path
        
    cPickle.dump(path_to_id_db, open(resource_prefix + '_path_to_id.p', 'wb'))
    cPickle.dump(id_to_path_db, open(resource_prefix + '_id_to_path.p', 'wb'))
    print 'Created path databases...'
    
    # Relations
    patterns_db = {}
    num_line = 0

    # Load the triplets file
    edges = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
    print 'Creating patterns.... '
    paths = set(paths)
    with gzip.open(triplets_file) as f:
        for line in f:
            try:
                x, y, path = line.strip().split('\t')
            except:
                print line
                continue

            # Frequent path
            if path in paths:
                x_id, y_id, path_id = term_to_id.get(x, -1), term_to_id.get(y, -1), path_to_id.get(path, -1)
                if x_id > -1 and y_id > -1 and path_id > -1:
                    edges[x_id][y_id][path_id] += 1

            num_line += 1
            if num_line % 1000000 == 0:
                print 'Processed ', num_line, ' lines.'

    for x in edges.keys():
        for y in edges[x].keys():
            patterns_db[str(x) + '###' + str(y)] = ','.join(
                [':'.join((str(p), str(val))) for (p, val) in edges[x][y].iteritems()])
    
    cPickle.dump(patterns_db, open(resource_prefix + '_patterns.p', 'wb'))
    print 'Done.............!'

if __name__ == '__main__':
    main()
