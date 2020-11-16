import torch
import numpy as np
from ikea import *

class Sampler():
    """
        Sampler to obtain k-way (k classes) n-shot (n per class) samples.
        label is the list of labels for all training samples.
        __iter__ returns indice of the sample to train on.
    """
    def __init__(self, label, n_batch, n_class, n_per_class, limit_class=False):
        self.n_batch = n_batch
        self.n_class = n_class
        self.n_per_class = n_per_class

        label = np.array(label)
        self.index_map = []

        #print(label, class_name)
        # experiemnt -> only 4 classes.
        for i in range(n_class):
            index = np.argwhere(label==i).reshape(-1)
            self.index_map.append(torch.from_numpy(index))

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for b in range(self.n_batch):
            batch = []
            classes = []
            classes = torch.randperm(len(self.index_map)) #[:self.n_class]
            #print("classes: ", len(classes))
            assert len(classes) == self.n_class
            for c in classes:
                ind_list = self.index_map[c]
                pos_list = torch.randperm(len(ind_list))[:self.n_per_class]
                batch.append(ind_list[pos_list])
            #print(batch)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

'''
def test_sampler():
    dataset = IkeaSet(TRAIN_MAT_PATH)
    print(len(dataset.labelSet))
    test_sampler = Sampler(dataset.label, dataset.final_classes, 10, 4, 6, limit_class=True)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler, num_workers=4, pin_memory=True)
    for i, batch in enumerate(test_loader, 1):
        print(np.array(batch[0]).shape)
        #print(i, batch)

if __name__ == "__main__":
    test_sampler()
'''