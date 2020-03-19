from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from train import BengaliDataset, get_augmenter, BengaliMixUpDataset, BengaliCutMixDataset, BengaliCutMixUpDataset


def debug_normal():

    train_csv = pd.read_csv('../../dataset/train.csv')

    dataset = BengaliDataset(Path('../../input/train_raw'), train_csv=train_csv,
                             is_aug=True, get_augmenter_func=get_augmenter)

    dir_save = Path('_debug')
    dir_save.mkdir(exist_ok=True)

    for i in range(20):
        (img_raw, img_square), label = dataset[0]

        img_raw = (img_raw.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_raw[0, :, :]).save(dir_save / f'img_raw{i}.png')

        img_square = (img_square.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_square[0, :, :]).save(dir_save / f'img_square{i}.png')


def debug_mixup():

    train_csv = pd.read_csv('../../dataset/train.csv')

    dataset = BengaliMixUpDataset(
        1.0,
        Path('../../input/train_raw'),
        train_csv=train_csv,
        is_aug=True,
        get_augmenter_func=get_augmenter
    )

    dir_save = Path('_debug')
    dir_save.mkdir(exist_ok=True)

    for i in range(20):
        (img_raw, img_square), label = dataset[0]

        img_raw = (img_raw.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_raw[0, :, :]).save(dir_save / f'img_raw{i}.png')

        img_square = (img_square.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_square[0, :, :]).save(dir_save / f'img_square{i}.png')


def debug_cutmix():

    train_csv = pd.read_csv('../../dataset/train.csv')

    dataset = BengaliCutMixDataset(
        1.0,
        Path('../../input/train_raw'),
        train_csv=train_csv,
        is_aug=True,
        get_augmenter_func=get_augmenter
    )

    dir_save = Path('_debug')
    dir_save.mkdir(exist_ok=True)

    for i in range(20):
        (img_raw, img_square), label = dataset[0]

        img_raw = (img_raw.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_raw[0, :, :]).save(dir_save / f'img_raw{i}.png')

        img_square = (img_square.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_square[0, :, :]).save(dir_save / f'img_square{i}.png')


def debug_cutmixup():

    train_csv = pd.read_csv('../../dataset/train.csv')

    dataset = BengaliCutMixUpDataset(
        0.5, 0.5,
        Path('../../input/train_raw'),
        train_csv=train_csv,
        is_aug=True,
        get_augmenter_func=get_augmenter
    )

    dir_save = Path('_debug')
    dir_save.mkdir(exist_ok=True)

    for i in range(20):
        (img_raw, img_square), label = dataset[0]

        img_raw = (img_raw.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_raw[0, :, :]).save(dir_save / f'img_raw{i}.png')

        img_square = (img_square.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_square[0, :, :]).save(dir_save / f'img_square{i}.png')


def main():

    # debug_normal()
    # debug_mixup()
    # debug_cutmix()
    debug_cutmixup()


if __name__ == '__main__':
    main()
