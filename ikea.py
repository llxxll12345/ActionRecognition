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
        self.final_classes = set()

        classes, class_cnt, clips, clip_label = load_data(mat_path)
        #print(class_cnt)
        
        for i in range (len(clips)):
            label = clip_label[i]
            if (class_cnt[label] >= 6):
                self.data.append(clips[i])
                self.label.append(clip_label[i])
                self.final_classes.add(clip_label[i])

        class_map = {}
        cnt = 0
        for c in self.final_classes:
            class_map[c] = cnt
            cnt += 1

        #print(class_map)
        self.label = [class_map[l] for l in self.label] 
        #print(self.labels)
        print(classes)
        print(self.final_classes)
        #print(self.final_classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        clip, label = self.data[i], self.label[i]
        return clip, label

'''
def test():
    dataset = IkeaSet(TRAIN_MAT_PATH)
    train_loader = DataLoader(dataset=dataset, num_workers=8, pin_memory=True)
    for i, batch in enumerate(train_loader, 1):
        print(i, batch[0].shape, batch[1])

if __name__ == "__main__":
    test()
'''