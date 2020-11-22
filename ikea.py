import os
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.io import loadmat
import numpy as np
# Folder: images_background
# Structure: root/language/character/image

TRAIN_MAT_PATH = 'train.mat'
TEST_MAT_PATH = 'test.mat'

def load_data(mat_path):
    db = loadmat(mat_path)["data"]
    #print(db.size)
    #print(db[0][0].shape)
    classes = {}
    class_cnt = {}
    clips = []
    clip_label = [] 
    ok = False
    for data in db:
        label = data[1][0]
        #print(label)
        clip = data[0]
        if label not in classes:
            classes[label] = len(classes) + 1
            class_cnt[classes[label] - 1] = 0
        clip = np.reshape(clip, [3, 224, 224, 16])
        clip = clip.astype(np.float32)
        #print(clip.dtype)
        clip /= 255.0
        #print(clip.shape)
        if not ok:
            #print(clip)
            ok = True
        clips.append(clip)
        clip_label.append(classes[label] - 1)
        class_cnt[classes[label] - 1] += 1
    return classes, class_cnt, clips, clip_label


class IkeaSet(Dataset):
    def __init__(self, mat_path):
        self.labelSet = set()
        self.label = []
        self.data = []
    

        classes, class_cnt, clips, clip_label = load_data(mat_path)
        self.label = clip_label
        self.data = clips
        self.classes = classes

        class_map = {}
        cnt = 0
        for c in self.classes:
            class_map[c] = cnt
            cnt += 1
    
        self.class_cnt = class_cnt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        clip, label = self.data[i], self.label[i]
        return clip, label


def test():
    dataset = IkeaSet(TRAIN_MAT_PATH)
    train_loader = DataLoader(dataset=dataset, num_workers=8, pin_memory=True)
    for i, batch in enumerate(train_loader, 1):
        print(i, batch[0].shape, batch[1])

if __name__ == "__main__":
    test()

