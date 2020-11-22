import cv2
import numpy as np
from scipy.io import loadmat, savemat
import os
import sys

def get_range(labels):
    last_label = []
    ranges = []
    start = 0
    cnt_range = {}
    #print(labels)
    for i in range(len(labels)):
        label = labels[i]
        #print(label)
        if last_label != label:
            if len(last_label) > 0:
                if last_label not in cnt_range:
                    cnt_range[last_label] = 0
                cnt_range[last_label]+=1
                ranges.append((start, i, last_label))
            start = i
            last_label = label
    #print(ranges)
    return ranges

data = {}

def pick_sample(action_range, video_path):
    global data
    video = cv2.VideoCapture(video_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frame_count)
    for r in action_range:
        start = r[0]
        end = r[1]
        label = r[2]
        if label not in data:
            data[label] = []

        if end - start <= 16: 
            continue
        delta = (end-start) // 20
        print(delta)
        for i in range(start + 8, end - 8, delta):
            clip = []
            for j in range(i - 8, i + 8):
                video.set(cv2.CAP_PROP_POS_FRAMES, j)
                success, frame = video.read()     
                if not success:
                    print("FAIL")
                    return
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                clip.append(frame)
            data[label].append(np.array(clip))

def main():
    for i in range(4,5):
        db = loadmat('IkeaClipsDB.mat', squeeze_me=True)['IkeaDB']
        print("Reading video ", i)

        rec = db[i]
        path_prefix = '/data/home/cherian/IkeaDataset/Frames/'
        video_name = rec['clip_path'][len(path_prefix):]
        video_path = os.path.join('videos', video_name) + '.MP4'
        print(video_path)

        labels = [act for act in rec['activity_labels']]
        ranges = get_range(labels)
        pick_sample(ranges, video_path)
        
    final_data = []
    for k, v in data.items():
        if len(v) > 20:
            print(k)
            v1 = np.random.choice(len(v), 20, replace=False)
            v = np.array(v)
            v = v[v1]
        for c in v:
            final_data.append((c, k))
    
    print(len(final_data))

    data_dict = {'data': final_data}
    savemat("train.mat", data_dict)

#main()

def split_train_test():
    db = loadmat("test.mat")["data"]
    classes = {}
    test_data = []
    train_data = []
    for data in db:
        label = data[1][0]
        #print(label)
        clip = data[0]
        if label not in classes:
            classes[label] = []
        classes[label].append(clip)
    for k, v in classes.items():
        n = int(len(v) * 0.2)
        vtrain = v[:-n]
        vtest = v[-n:]
        for vt in vtrain:
            train_data.append((vt, k))
        for vi in vtest:
            test_data.append((vi, k))
    
    train_dict = {'data': train_data}
    savemat("train1.mat", train_dict)
    test_dict = {'data': test_data}
    savemat("test1.mat", test_dict)


def test_read_data():
    db = loadmat("test1.mat")["data"]
    print(db.size)
    print(db[0][0].shape)
    classes = {}
    class_cnt = {}
    clips = []
    for data in db:
        label = data[1][0]
        #print(label)
        clip = data[0]
        if label not in classes:
            classes[label] = len(classes) + 1
            class_cnt[label] = 0
        clips.append((clip, classes[label]))
        class_cnt[label] += 1
    print(classes)            
    print(class_cnt)
    print(len(clips), clips[0][0].shape)


split_train_test()
test_read_data()