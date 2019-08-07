# coding:utf8
# author: Gofinge

import warnings


class DefaultConfig(object):
    model = 'densenet201'

    data_root = '/Users/gofinge/Documents/DATA/'
    train_data_list = './data/trainSet.csv'
    load_model_path = None
    classes = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Consolidation', 'Edema']

    batch_size = 10  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 2  # how many workers for loading data
    check_freq = 2000  # check model performance every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 3
    lr = 0.0001  # initial learning rate
    betas = (0.9, 0.999)
    eps = 1e-08
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5


def parse(self, kwargs):
    """
     update Config through kwargs
    """
    if kwargs:
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
