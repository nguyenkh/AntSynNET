from __future__ import print_function
import argparse
from collections import OrderedDict
import sys
import time
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sklearn.metrics import precision_recall_fscore_support
import common

SEED = 123089 # Could change to any number
numpy.random.seed(SEED)

def main():
    """
    AntSynNET model
    Usage:
        python train_ant_syn_net.py -corpus <corpus_prefix> -data <dataset_prefix> -emb <embeddings_file>
                                    -model <model_name> -iter <iteration>
    <corpus_prefix>: the prefix of corpus
    <dataset_prefix>: the prefix of dataset
    <embeddings_file>: the embeddings file
    <model_name>: 1 for training combined model or 0 for training pattern-based model
    <iteration>: the number of iteration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus', type=str)
    parser.add_argument('-data', type=str)
    parser.add_argument('-emb', action='store', default=None, dest='emb_file')
    parser.add_argument('-model', type=int, default=1)
    parser.add_argument('-iter', type=int)
    args = parser.parse_args()
    # See AntSynNET function for all possible parameters and their definitions.
    AntSynNET(corpus_prefix=args.corpus, 
              dataset_prefix=args.data,
              embeddings_file=args.emb_file,
              model=args.model,
              epochs=args.iter)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def _p(pp, name):
    return '%s_%s' % (pp, name)

def init_params(options, w2v=None):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    if w2v is None:
        randn_lemma = numpy.random.rand(options['n_words'],
                              options['dim_lemma'])
        params['Wemb'] = (0.01 * randn_lemma).astype(config.floatX)
    else:
        params['Wemb'] = w2v.astype(config.floatX)
        
    randn_pos = numpy.random.rand(options['n_pos'],
                              options['dim_pos'])
    params['POS'] = (0.01 * randn_pos).astype(config.floatX)
    
    randn_dep = numpy.random.rand(options['n_dep'],
                              options['dim_dep'])
    params['DEP'] = (0.01 * randn_dep).astype(config.floatX)
    
    randn_dist = numpy.random.rand(options['n_dist'],
                              options['dim_dist'])
    params['DIST'] = (0.01 * randn_dist).astype(config.floatX)
        
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_classifier'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params

def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def get_layer(name):
    fns = layers[name]
    return fns

def init_tparams(params, ):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)

def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0][-1]

layers = {'lstm': (param_init_lstm, lstm_layer)}

def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    grad_updates = gsup

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    params_updates = pup

    return grad_updates, params_updates

def adadelta(lr, tparams, grads):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    grad_updates=zgup + rg2up
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    params_updates = ru2up + param_up

    return grad_updates, params_updates

def rmsprop(lr, tparams, grads):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    grad_updates = zgup + rgup + rg2up

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    
    params_updates = updir_new + param_up

    return grad_updates, params_updates

def pred_error(f_pred, preprocess_data, data, iterator, pair_vectors=None):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    preprocess_data: usual preprocess_data for that dataset.
    """
    preds = []
    targets = []
    for valid_index in iterator:
        l, p, g, d, mask, c = preprocess_data(data[0][valid_index])
        if pair_vectors is not None:
            source_term, target_term = pair_vectors[valid_index]
            pred = f_pred(l, p, g, d, mask, c, source_term, target_term)
        else:
            pred = f_pred(l, p, g, d, mask, c)
        preds.extend(pred)
        targets.append(data[1][valid_index])
           
    preds = numpy.array(preds)
    targets = numpy.array(targets)  
    valid_err = (preds == targets).sum()
    
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def predict(f_pred, preprocess_data, data, iterator, pair_vectors=None):
    preds = []
    targets = []
    for valid_index in iterator:
        l, p, g, d, mask, c = preprocess_data(data[0][valid_index])
        if pair_vectors is not None:
            source_term, target_term = pair_vectors[valid_index]
            pred = f_pred(l, p, g, d, mask, c, source_term, target_term)
        else:
            pred = f_pred(l, p, g, d, mask, c)
        preds.extend(pred)
        targets.append(data[1][valid_index])
        
    return preds, targets

def AntSynNET(
    corpus_prefix, # The prefix of corpus.
    dataset_prefix, # The prefix of dataset.
    embeddings_file=None, # File of pre-trained word embeddings.
    model=1, # 1 for training combined model or 0 for training pattern-based model.
    dim_lemma=100, # The number dimension of lemma.
    dim_pos=10, # The number dimension of POS.
    dim_dep=10, # The number dimension of dependency label.
    dim_dist=10, # The number dimension of distance label.
    epochs=40,  # The maximum number of epoch to run
    dispFreq=1000,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommended (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    save_model='AntSynNET_model.npz',  # The parameters of model will be saved there
    maxlen=12,  # Sequence longer then this get ignored
    lrate=0.0001, # Learning rate for sgd (not used for adadelta and rmsprop)
    noise_std=0.5, # Dropout rate
    use_dropout=True,  # if False slightly faster, but worst test error
    reload_model=None,  # Path to a saved model we want to start from.
):
    
    # Model options
    model_options = locals().copy()

    load_data, preprocess_instance = common.load_data, common.preprocess_instance

    # Loads data
    pre_trained_vecs, train, valid, test, keys, lemma_dict, pos_dict, dep_dict, dist_dict = \
        load_data(corpus_prefix, dataset_prefix, embeddings_file)
        
    X_train, y_train = train
    X_test, y_test = test
    X_valid, y_valid = valid
    
    train_key_pairs, test_key_pairs, valid_key_pairs = keys
    
    kf_valid = numpy.random.permutation(len(X_valid))
    kf_test = numpy.random.permutation(len(X_test))

    print("%d train examples" % len(y_train))
    print("%d valid examples" % len(y_valid))
    print("%d test examples" % len(y_test))
        
    ydim = numpy.max(train[1]) + 1
    model_options['n_words'] = len(lemma_dict)
    model_options['n_pos'] = len(pos_dict)
    model_options['n_dep'] = len(dep_dict)
    model_options['n_dist'] = len(dist_dict)
    
    if pre_trained_vecs is not None:
        model_options['dim_lemma'] = len(pre_trained_vecs[0])
        
    model_options['dim_proj'] = model_options['dim_lemma'] + dim_pos + dim_dep + dim_dist
    
    if model_options['model']==1:
        model_options['dim_classifier'] = 2 * model_options['dim_lemma'] + model_options['dim_proj']
    elif model_options['model']==0:
        model_options['dim_classifier'] = model_options['dim_proj']
    else:
        print('1 for training combined model or 0 for training pattern-based model')
        
    model_options['ydim'] = ydim
    
    print("model options", model_options)

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options, pre_trained_vecs)

    if reload_model is not None:
        load_params(reload_model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)
    
    trng = RandomStreams(SEED)
    use_noise = theano.shared(numpy_floatX(0.))

    l = tensor.matrix('lemma', dtype='int64')
    p = tensor.matrix('pos', dtype='int64')
    g = tensor.matrix('dep', dtype='int64')
    d = tensor.matrix('dist', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    c = tensor.matrix('freq', dtype='int64')
    y = tensor.vector('y', dtype='int64')
    lr = tensor.scalar(name='lr')
    
    if model_options['model']==1:
        s_term = tensor.scalar('source term', dtype='int64')
        t_term = tensor.scalar('target term', dtype='int64')

    zero_vec_lemma = tensor.vector()
    zero_vec_pos = tensor.vector()
    zero_vec_dep = tensor.vector()
    zero_vec_dist = tensor.vector()
    zero_lemma = numpy.zeros(model_options['dim_lemma'])
    zero_pos = numpy.zeros(model_options['dim_pos'])
    zero_dep = numpy.zeros(model_options['dim_dep'])
    zero_dist = numpy.zeros(model_options['dim_dist'])    
    
    n_timesteps = l.shape[0]
    n_samples = l.shape[1]
    
    lemma = tparams['Wemb'][l.flatten()].reshape([n_timesteps, n_samples, model_options['dim_lemma']])
    pos = tparams['POS'][p.flatten()].reshape([n_timesteps, n_samples, model_options['dim_pos']])
    dep = tparams['DEP'][g.flatten()].reshape([n_timesteps, n_samples, model_options['dim_dep']])
    dist = tparams['DIST'][d.flatten()].reshape([n_timesteps, n_samples, model_options['dim_dist']])

    if model_options['use_dropout']:
        lemma = dropout_layer(lemma, use_noise, trng)
        pos = dropout_layer(pos, use_noise, trng)
        dep = dropout_layer(dep, use_noise, trng)
        dist = dropout_layer(dist, use_noise, trng)
 
    emb = tensor.concatenate([lemma,pos,dep,dist], axis=2)
    
    proj = get_layer(model_options['encoder'])[1](tparams, emb, model_options,
                                                  prefix=model_options['encoder'], mask=mask)

    if model_options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    proj = tensor.dot(tensor.cast(c, 'float32'), proj)
    proj = tensor.sum(proj,axis=0) / tensor.sum(c).astype(config.floatX)
    
    if model_options['model']==1:
        source_term_emb = tparams['Wemb'][s_term]
        target_term_emb = tparams['Wemb'][t_term]
        proj = tensor.concatenate([source_term_emb, proj, target_term_emb])
        
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    if model_options['model']==1:
        f_pred_prob = theano.function([l, p, g, d, mask, c, s_term, t_term], pred, 
                                      name='f_pred_prob', on_unused_input='ignore')
        f_pred = theano.function([l, p, g, d, mask, c, s_term, t_term], pred.argmax(axis=1), 
                                 name='f_pred', on_unused_input='ignore')
    else:
        f_pred_prob = theano.function([l, p, g, d, mask, c], pred, 
                                      name='f_pred_prob', on_unused_input='ignore')
        f_pred = theano.function([l, p, g, d, mask, c], pred.argmax(axis=1), 
                                 name='f_pred', on_unused_input='ignore')
    
    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    #n_samples = 1 corresponding to batch-size = 1
    cost = -tensor.log(pred[tensor.arange(1), y] + off).mean()
    
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    
    grad_updates, params_updates = optimizer(lr, tparams, grads)
    
    if model_options['model']==1:    
        f_grad_shared = theano.function([l, p, g, d, mask, c, s_term, t_term, y], cost, updates=grad_updates,
                                    name='adadelta_f_grad_shared', on_unused_input='ignore')
    else:
        f_grad_shared = theano.function([l, p, g, d, mask, c, y], cost, updates=grad_updates,
                                    name='adadelta_f_grad_shared', on_unused_input='ignore')
        
    f_update = theano.function([lr], [], updates=params_updates, on_unused_input='ignore',
                                name='adadelta_f_update')
    
    set_zero_lemma = theano.function([zero_vec_lemma], 
                                     updates=[(tparams['Wemb'], tensor.set_subtensor(tparams['Wemb'][0,:], zero_vec_lemma))],
                                     allow_input_downcast=True)
    set_zero_pos = theano.function([zero_vec_pos], 
                                   updates=[(tparams['POS'], tensor.set_subtensor(tparams['POS'][0,:], zero_vec_pos))],
                                   allow_input_downcast=True)
    set_zero_dep = theano.function([zero_vec_dep], 
                                   updates=[(tparams['DEP'], tensor.set_subtensor(tparams['DEP'][0,:], zero_vec_dep))],
                                   allow_input_downcast=True) 
    set_zero_dist = theano.function([zero_vec_dist], 
                                    updates=[(tparams['DIST'], tensor.set_subtensor(tparams['DIST'][0,:], zero_vec_dist))],
                                    allow_input_downcast=True) 
    
    print('Training.......')
        
    uidx = 0  # the number of update done
    start_time = time.time()
    try:
        for eidx in range(epochs):
            n_samples = 0
            dispLoss = 0.
            # Get new shuffled index for the training set.
            kf_train = numpy.random.permutation(len(X_train))
                
            for train_idx in kf_train:
                instance = X_train[train_idx]
                l, p, g, d, mask, c = preprocess_instance(instance)
                source_term, target_term = train_key_pairs[train_idx]
                label = [y_train[train_idx]]
                
                uidx += 1
                use_noise.set_value(noise_std)
                
                if model_options['model']==1:
                    cost = f_grad_shared(l, p, g, d, mask, c, source_term, target_term, label)
                else:
                    cost = f_grad_shared(l, p, g, d, mask, c, label)
                    
                f_update(lrate)

                set_zero_lemma(zero_lemma)
                set_zero_pos(zero_pos)
                set_zero_dep(zero_dep)
                set_zero_dist(zero_dist)
                                                          
                dispLoss += cost
                n_samples += 1

                if numpy.mod(uidx, dispFreq) == 0:
                    cost = dispLoss / n_samples
                    use_noise.set_value(0.)
                    if model_options['model']==1:
                        valid_err = pred_error(f_pred, preprocess_instance, valid, kf_valid, valid_key_pairs)
                    else:
                        valid_err = pred_error(f_pred, preprocess_instance, valid, kf_valid)
                    print('Epoch: %d, Update: %d, Cost: %f, Valid: %.3f' %(eidx, uidx, cost, valid_err))
                    
            print('Seen %d samples' % n_samples)

    except KeyboardInterrupt:
        print("Training interupted")
        
    use_noise.set_value(0.)
    if model_options['model']==1:
        preds, targets = predict(f_pred, preprocess_instance, test, kf_test, test_key_pairs) 
    else:
        preds, targets = predict(f_pred, preprocess_instance, test, kf_test)
        
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
    print ('Precision: %.3f, Recall: %.3f, F1: %.3f' % (p, r, f1))
    
    save_params = unzip(tparams)
    if save_model is not None:
        numpy.savez(save_model, preds=preds, targets=targets, p=p, r=r, f1=f1, **save_params)
    
    end_time = time.time()
    
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return p, r, f1

if __name__ == '__main__':
    main()

