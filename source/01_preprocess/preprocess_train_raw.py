from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

HEIGHT = 137
WIDTH = 236
SIZE = 128


def main():

    dir_save = Path('../../input/train_raw')
    dir_save.mkdir(exist_ok=True)

    target_files = [
        '../../dataset/train_image_data_0.parquet',
        '../../dataset/train_image_data_1.parquet',
        '../../dataset/train_image_data_2.parquet',
        '../../dataset/train_image_data_3.parquet'
    ]

    for fname in target_files:

        df = pd.read_parquet(fname)

        # the input is inverted
        data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            name = row['image_id']

            # normalize each image by its max val
            img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)

            cv2.imwrite(str(dir_save / f'./{name}.png'), img)


if __name__ == '__main__':
    main()
