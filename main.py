# coding: utf8
# author: Gofinge

import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from data.dataset import ChestXrayDataSet
import models
from config import opt


def test(**kwargs):
    opt.parse(kwargs)
    # configure model
    model = _generate_model()

    # data
    test_data = ChestXrayDataSet(opt.data_root, opt.train_data_list, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    total_batch = int(len(test_data) / opt.batch_size)
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()

    if opt.use_gpu:
        gt = gt.cuda()
        pred = pred.cuda()

    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(test_dataloader), total=total_batch)
        for i, (data, label) in bar:
            bs, n_crops, c, h, w = data.size()
            ipt = inp.view(-1, c, h, w)
            inp = data.clone().detach()
            target = label.clone().detach()
            if opt.use_gpu:
                inp = inp.cuda()
                target = target.cuda()

            output = model(inp)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            gt = torch.cat((gt, target), 0)
            pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(len(opt.classes)):
        print('The AUROC of {} is {}'.format(opt.classes[i], AUROCs[i]))

    _write_csv(AUROCs, opt.result_file)


def train(**kwargs):
    opt.parse(kwargs)

    # step1: configure model
    model = _generate_model()

    # step2: data
    train_data = ChestXrayDataSet(opt.data_root, opt.train_data_list, mode='train')
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(train_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = torch.nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=opt.betas,
                                 eps=opt.eps, weight_decay=opt.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')

    # step4: meters
    loss_mean_min = 1e100

    # train
    print('\n---------------------------------')
    print(' ٩( ᐛ )و - Start training ......')
    print('---------------------------------\n')
    for epoch in range(opt.max_epoch):
        print('(๑•̀ㅂ•́)و✧ - Epoch', epoch + 1, ':')
        model.train()
        total_batch = int(len(train_data) / opt.batch_size)

        bar = tqdm(enumerate(train_dataloader), total=total_batch)
        for i, (data, label) in bar:
            # train model
            torch.set_grad_enabled(True)
            inp = data.clone().detach().requires_grad_(True)
            target = label.clone().detach()
            if opt.use_gpu:
                inp = inp.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = model(inp)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            bar.set_postfix_str('loss: %.5s' % loss.item())

        loss_mean = _val(model, val_dataloader, criterion, total_batch)
        time_end = time.strftime('%m%d_%H%M%S')
        scheduler.step(loss_mean)
        if loss_mean_min > loss_mean:
            loss_mean_min = loss_mean
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       './checkpoints/m_' + time_end + '.pth.tar')
            print('(˃̶ᗜ˂̶)✩ Epoch [' + str(epoch + 1) + '] [save] [' + time_end + '] loss= ' + str(loss_mean))
        else:
            print('(இωஇ) Epoch [' + str(epoch + 1) + '] [----] [' + time_end + '] loss= ' + str(loss_mean))
        print('------------------------------------------------------------------------------------\n')


def _val(model, dataloader, criterion, total_batch):
    model.eval()
    counter = 0
    loss_sum = 0

    with torch.no_grad():
        bar = tqdm(enumerate(dataloader), total=total_batch)
        for i, (data, label) in bar:
            inp = data.clone().detach()
            target = label.clone().detach()
            if opt.use_gpu:
                inp = inp.cuda()
                target = target.cuda()

            output = model(inp)

            loss = criterion(output, target)
            loss_sum += loss.item()
            counter += 1
            bar.set_postfix_str('loss: %.5s' % loss.item())

    loss_mean = loss_sum / counter
    return loss_mean


def _generate_model():
    model = getattr(models, opt.model)(len(opt.classes))

    if opt.load_model_path:
        load_model_path = os.path.join('./checkpoint_pth', opt.load_model_path)
        assert os.path.isfile(load_model_path), 'No checkpoint found.'
        print('Loading checkpoint......')
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done')

    if opt.use_gpu:
        model.cuda()
    model = torch.nn.DataParallel(model)
    return model


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(len(opt.classes)):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def _write_csv(results, file_name):
    df = pd.DataFrame(results)
    df.to_csv(file_name, sep=' ')


if __name__ == '__main__':
    train()
