import os
import time
import json
import random
import argparse
from pathlib import Path
from distutils.util import strtobool
from logging import getLogger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler
from albumentations import ElasticTransform, Compose, RandomGamma

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Recall
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar, CosineAnnealingScheduler

from submit import build_model, BengaliDataset, NUM_GRAPHEME_ROOT, NUM_VOWEL, NUM_CONSONANT
from lib.log import init_logger
from lib.sampler import SequentialSampler, BalancedSampler


class BengaliMixUpDataset(BengaliDataset):

    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def __getitem__(self, idx):
        (img_raw1, img_square1), (_, one_hot_targets1) = super().__getitem__(idx)

        lam = np.random.beta(self.alpha, self.alpha)

        (img_raw2, img_square2), (_, one_hot_targets2) = super().__getitem__(random.randrange(0, len(self)))

        img_raw = lam * img_raw1 + (1 - lam) * img_raw2
        img_square = lam * img_square1 + (1 - lam) * img_square2
        mixed_up_targets = lam * one_hot_targets1 + (1 - lam) * one_hot_targets2

        return (img_raw, img_square), (_, mixed_up_targets)


class BengaliCutMixDataset(BengaliDataset):

    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    @staticmethod
    def get_bbox(W, H, cut_rat):

        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __getitem__(self, idx):

        (img_raw1, img_square1), (_, one_hot_targets1) = super().__getitem__(idx)

        lam = np.random.beta(self.alpha, self.alpha)

        (img_raw2, img_square2), (_, one_hot_targets2) = super().__getitem__(random.randrange(0, len(self)))

        cut_rat = np.sqrt(1. - lam)

        bbx1, bby1, bbx2, bby2 = self.get_bbox(img_raw1.shape[2], img_raw1.shape[1], cut_rat)
        img_raw1[:, bby1:bby2, bbx1:bbx2] = img_raw2[:, bby1:bby2, bbx1:bbx2]

        bbx1, bby1, bbx2, bby2 = self.get_bbox(img_square1.shape[2], img_square1.shape[1], cut_rat)
        img_square1[:, bby1:bby2, bbx1:bbx2] = img_square2[:, bby1:bby2, bbx1:bbx2]

        mixed_up_targets = lam * one_hot_targets1 + (1 - lam) * one_hot_targets2

        return (img_raw1, img_square1), (_, mixed_up_targets)


class BengaliCutMixUpDataset(BengaliCutMixDataset):

    def __init__(self, alpha1, alpha2, *args, **kwargs):
        super().__init__(alpha2, *args, **kwargs)
        self.alpha = alpha1

    def __getitem__(self, idx):
        (img_raw1, img_square1), (_, one_hot_targets1) = super().__getitem__(idx)

        lam = np.random.beta(self.alpha, self.alpha)

        (img_raw2, img_square2), (_, one_hot_targets2) = super().__getitem__(random.randrange(0, len(self)))

        img_raw = lam * img_raw1 + (1 - lam) * img_raw2
        img_square = lam * img_square1 + (1 - lam) * img_square2
        mixed_up_targets = lam * one_hot_targets1 + (1 - lam) * one_hot_targets2

        return (img_raw, img_square), (_, mixed_up_targets)


class BengaliLoss(_Loss):

    def forward(self, input, target):

        targets, one_hot_targets = target

        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        loss1 = criterion.forward(
            F.log_softmax(input[:, :NUM_GRAPHEME_ROOT], dim=1),
            one_hot_targets[:, :NUM_GRAPHEME_ROOT]
        )

        loss2 = criterion.forward(
            F.log_softmax(input[:, NUM_GRAPHEME_ROOT:NUM_GRAPHEME_ROOT+NUM_VOWEL], dim=1),
            one_hot_targets[:, NUM_GRAPHEME_ROOT:NUM_GRAPHEME_ROOT+NUM_VOWEL]
        )

        loss3 = criterion.forward(
            F.log_softmax(input[:, NUM_GRAPHEME_ROOT+NUM_VOWEL:], dim=1),
            one_hot_targets[:, NUM_GRAPHEME_ROOT+NUM_VOWEL:]
        )

        loss = loss1 + loss2 + loss3
        return loss


def eval_print(target, epoch, metrics):

    logger = getLogger('root')

    accuracy_gr = metrics['accuracy_gr']
    accuracy_vd = metrics['accuracy_vd']
    accuracy_cd = metrics['accuracy_cd']
    recall_gr = metrics['recall_gr']
    recall_vd = metrics['recall_vd']
    recall_cd = metrics['recall_cd']
    ave_recall = metrics['ave_recall']

    logger.info(
        f'{target} Scores - Epoch:{epoch:2d}  '
        f'Acc [(GR): {accuracy_gr:.3f} (VD): {accuracy_vd:.3f} (CD): {accuracy_cd:.3f}]  '
        f'Recall [(GR): {recall_gr:.3f} (VD): {recall_vd:.3f} (CD): {recall_cd:.3f} (Ave): {ave_recall:.3f}]'
    )


def score_function(engine):
    score = engine.state.metrics['ave_recall']
    return score


def get_augmenter(p=1.0):
    return Compose([
        ElasticTransform(p=0.8, alpha_affine=10, border_mode=cv2.BORDER_REPLICATE),
        RandomGamma(p=0.8, gamma_limit=(50, 150)),
        ], p=p)


def train(model_name: str, dir_dataset: Path, dir_model: Path, img_type: str,
          architecture: str, epoch: int, batch_size: int, multi_sample_aug: str,
          seed: int, debug: bool):

    logger = getLogger('root')

    tic = time.time()

    random.seed(seed)
    np.random.seed(seed)

    train_csv = pd.read_csv(dir_dataset / 'train.csv')

    dir_images = Path('../../input/train_raw')

    train_images_all = sorted(os.listdir(str(dir_images)))
    random.shuffle(train_images_all)

    num_train = int(len(train_images_all) * 0.9)
    train_images = train_images_all[:num_train]
    valid_images = train_images_all[num_train:]

    if debug:
        train_images = train_images[:4000]
        valid_images = valid_images[:2000]

    logger.info('train images: {}'.format(train_images[:5]))
    logger.info('valid images: {}'.format(valid_images[:5]))

    logger.info('==> prepare dataset')
    logger.info(f'==> use {multi_sample_aug}')

    if multi_sample_aug == 'mixup':
        train_mixup_dataset = BengaliMixUpDataset(
            1.0, dir_images,
            train_csv=train_csv, list_images=train_images,
            is_aug=True, get_augmenter_func=get_augmenter
        )
    elif multi_sample_aug == 'cutmix':
        train_mixup_dataset = BengaliCutMixDataset(
            1.0, dir_images,
            train_csv=train_csv, list_images=train_images,
            is_aug=True, get_augmenter_func=get_augmenter
        )
    elif multi_sample_aug == 'cutmixup':
        train_mixup_dataset = BengaliCutMixUpDataset(
            0.5, 0.5, dir_images,
            train_csv=train_csv, list_images=train_images,
            is_aug=True, get_augmenter_func=get_augmenter
        )
    else:
        raise ValueError('Unknown Augmentation')

    train_dataset = BengaliDataset(
        dir_images,
        train_csv=train_csv, list_images=train_images,
        is_aug=False
    )
    valid_dataset = BengaliDataset(
        dir_images,
        train_csv=train_csv, list_images=valid_images,
        is_aug=False
    )

    logger.info('==> create data loader')
    train_loader = DataLoader(
        train_mixup_dataset, num_workers=7,
        batch_sampler=BalancedSampler(train_csv, train_images, len(train_images) // batch_size, batch_size)
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=7, shuffle=False
    )
    eval_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=7, shuffle=False,
        sampler=SequentialSampler(train_dataset, num_samples=len(valid_images))
    )

    logger.info('==> build model')
    model = build_model(architecture, img_type, pretrained=True)
    model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_f = BengaliLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = create_supervised_trainer(model, optimizer, loss_f, device=device)

    ProgressBar(persist=True, desc='Train').attach(trainer)

    def extract_grapheme_root(output):
        y_pred, (y, _) = output
        return y_pred[:, :NUM_GRAPHEME_ROOT], y[:, 0]

    def extract_vowel_diacritic(output):
        y_pred, (y, _) = output
        return y_pred[:, NUM_GRAPHEME_ROOT:NUM_GRAPHEME_ROOT+NUM_VOWEL], y[:, 1]

    def extract_consonant_diacritic(output):
        y_pred, (y, _) = output
        return y_pred[:, NUM_GRAPHEME_ROOT+NUM_VOWEL:], y[:, 2]

    metrics = {
        'accuracy_gr': Accuracy(extract_grapheme_root),
        'accuracy_vd': Accuracy(extract_vowel_diacritic),
        'accuracy_cd': Accuracy(extract_consonant_diacritic),
        'recall_gr': Recall(extract_grapheme_root, average=True),
        'recall_vd': Recall(extract_vowel_diacritic, average=True),
        'recall_cd': Recall(extract_consonant_diacritic, average=True),
        'ave_recall': 0.5 * Recall(extract_grapheme_root, average=True) +
                      0.25 * Recall(extract_vowel_diacritic, average=True) +
                      0.25 * Recall(extract_consonant_diacritic, average=True)
    }

    evaluator_train = create_supervised_evaluator(model, metrics=metrics, device=device)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    ProgressBar(persist=True, desc='Train Evaluation').attach(evaluator_train)
    ProgressBar(persist=True, desc='Valid Evaluation').attach(evaluator)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator_train.run(eval_train_loader)
        eval_print('Train', trainer.state.epoch, evaluator_train.state.metrics)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        eval_print('Valid', trainer.state.epoch, evaluator.state.metrics)

    handler = ModelCheckpoint(
        dirname=dir_model,
        filename_prefix=f'temp',
        create_dir=True,
        require_empty=False,
        n_saved=1,
        score_function=score_function
    )
    evaluator.add_event_handler(Events.COMPLETED, handler, {'model': model})

    lr_handler = CosineAnnealingScheduler(
        optimizer, 'lr', 1e-3, 1e-7, len(train_loader) * epoch, save_history=True)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_handler)

    trainer.run(train_loader, max_epochs=epoch)

    saved_model = handler._saved[0][1][0]
    model_new_name = str(dir_model / f'{model_name}.pth')
    logger.info(f'rename model {saved_model} to {model_new_name}')
    os.rename(saved_model, model_new_name)

    plt.plot(trainer.state.param_history['lr'])
    plt.xlabel('batch')
    plt.ylabel('learning rate')
    plt.savefig('_log/learning_rate.png')
    plt.close()

    model_conf = {
        'model_name': model_name,
        'architecture': architecture,
        'seed': seed,
        'img_type': img_type,
        'multi_sample_aug': multi_sample_aug
    }

    with open(str(params.dir_model / f'{model_name}.json'), 'w') as f:
        json.dump(model_conf, f, indent=2)

    elapsed_time = time.time() - tic
    logger.info(f'elapsed time: {elapsed_time / 60.0:.1f} [min]')


def main():

    if params.debug:
        logger = init_logger('_log/train_debug.log', level=10)
    else:
        logger = init_logger(f'_log/train_{params.model_name}.log', level=20)

    logger.info(f'torch version {torch.__version__}')
    logger.info(f'numpy version {np.__version__}')
    logger.info(f'pandas version {pd.__version__}')

    logger.info(vars(params))

    train(**vars(params))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='model_01', type=str)

    parser.add_argument('--dir_dataset', default=Path('../../dataset'), type=Path)

    parser.add_argument('--dir_model', default=Path('_model'), type=Path)

    parser.add_argument('--img_type', default='square_crop', type=str,
                        choices=['square_crop', 'square_resize', 'raw']
                        )

    parser.add_argument('--architecture', default='resnet34', type=str,
                        choices=['resnet34', 'resnet50', 'densenet121']
                        )

    parser.add_argument('--epoch', default=5, type=int)

    parser.add_argument('--batch_size', default=32, type=int)

    parser.add_argument('--multi_sample_aug', default='mixup', type=str,
                        choices=['mixup', 'cutmix', 'cutmixup']
                        )

    parser.add_argument('--debug', default=True, type=strtobool)

    parser.add_argument('--seed', default=1048, type=int)

    params = parser.parse_args()

    main()
