# coding:utf8
# author: Gofinge

import warnings


class DefaultConfig(object):
    model = 'densenet121'

    data_root = '/Users/gofinge/Documents/DATA/'
    train_data_list = './data/trainSet.csv'
    valid_data_list = './data/validSet.csv'
    test_data_list = './data/testSet.csv'
    load_model_path = None
    classes = ['Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Consolidation', 'Edema', 'Pneumonia']

    batch_size = 16  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 2  # how many workers for loading data
    check_freq = 2000  # check model performance every N batch

    result_file = 'result.csv'

    max_epoch = 5
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
            for k in kwargs:
                if not hasattr(self, k):
                    warnings.warn("Warning: opt has not attribut %s" % k)
                setattr(self, k, kwargs[k])

            # print('user config:')
            # for k in self.__class__.__dict__:
            #     if not k.startswith('__'):
            #         print(k, getattr(self, k))


opt = DefaultConfig()

