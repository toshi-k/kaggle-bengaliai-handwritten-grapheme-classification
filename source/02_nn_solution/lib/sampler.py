import random
from pathlib import Path
from torch.utils.data import Sampler


class SequentialSampler(Sampler):

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self.len = min(len(self.data_source), num_samples)

    def __iter__(self):
        return iter(range(self.len))

    def __len__(self):
        return self.len


class BalancedSampler(Sampler):

    def __init__(self, train, list_images, num_iter, batch_size):
        self.num_iter = num_iter
        self.batch_size = batch_size

        list_images = [Path(s).stem for s in list_images]
        dataset = train.query('image_id in @list_images')

        self.list_groups = sorted(list(set(dataset['grapheme_root'])))
        self.group_train = dataset.groupby('grapheme_root')

        self._i = 0

    def get_sample(self):
        list_imgs = list()
        target_groups = random.choices(self.list_groups, k=self.batch_size)
        for g in target_groups:
            group_df = self.group_train.get_group(g)
            img_id = random.choice(group_df['image_id'].tolist())
            list_imgs.append(f'{img_id}.png')
        return list_imgs

    def __len__(self):
        return self.num_iter

    def __iter__(self):

        while self._i < self.num_iter:
            yield self.get_sample()
            self._i += 1

        self._i = 0
