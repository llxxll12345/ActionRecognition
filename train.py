import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sampler import Sampler
from model import SimpleModel
import torch.nn as nn
from utils import *
from ikea import *
import datetime
import os

def save_model(model, name, save_path):
    torch.save(model.state_dict(), os.path.join(save_path, name+'.pth'))

def load_model(model, name, save_path):
    path = os.path.join(save_path, 'model.pth')
    if os.path.exists(os.path.join(save_path, 'model.pth')):
        print('load from previous model')
        pre_model = torch.load(path)
        model.load_state_dict(pre_model,strict=False)
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    renew_path(args.save)
    
    train_set = IkeaSet(TRAIN_MAT_PATH)
    train_loader = DataLoader(train_set, num_workers=4, pin_memory=True)

    test_set = IkeaSet(TEST_MAT_PATH)
    test_loader = DataLoader(test_set, num_workers=4, pin_memory=True)

    model = SimpleModel(n_class=len(train_set.classes)).to(device)
    model = load_model(model, 'model', args.save)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # learing rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = F.cross_entropy
    
    # training log
    training_log = {}
    training_log['args'] = []
    training_log['train_loss'] = []
    training_log['val_loss'] = []
    training_log['train_acc'] = []
    training_log['val_acc'] = []
    training_log['max_acc'] = 0.0

    for epoch in range(1, args.epoch + 1):
        time_a = datetime.datetime.now()
        model.train()
        average_loss = 0
        average_accuracy = 0
        for i, batch in enumerate(train_loader, 1):
            batch[0].to(device), batch[1].to(device)
            
            label = batch[1]
            label = label.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
            
            pred = model(batch[0])

            loss = loss_fn(pred, label)
            print(pred)
            print(torch.argmax(pred, dim=1), label)
            acc = get_accuracy(label, pred)
            if i % 20 == 0:
                print('epoch{}, {}/{}, lost={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))
            average_loss = update_avg(i + 1, average_loss, loss.item())
            average_accuracy = update_avg(i + 1, average_accuracy, acc)

            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            embedding = None
            loss = None
            distanece = None
        
        model.eval()
        average_loss_val = 0
        average_accuracy_val = 0

        # evaluate after epoch.
        with torch.no_grad():
            for i, batch in enumerate(test_loader, 1):
                batch[0].to(device), batch[1].to(device)

                label = batch[1]
                #label = query_y.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
                label = label.type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
                
                pred = model(batch[0])
                
                loss = loss_fn(pred, label)
                acc = get_accuracy(label, pred)
                average_loss_val = update_avg(i + 1, average_loss_val, loss.item())
                average_accuracy_val = update_avg(i + 1, average_accuracy_val, acc)

                embedding = None
                loss = None
                distanece = None

        print("epoch {} validation: loss={:4f} acc={:4f}".format(epoch, average_loss, average_accuracy))
        if average_accuracy > training_log['max_acc']:
            training_log['max_acc'] = acc
            save_model(model, 'max-acc', args.save)

        training_log['train_loss'].append(average_loss)
        training_log['train_acc'].append(average_accuracy)
        training_log['val_loss'].append(average_loss_val)
        training_log['val_acc'].append(average_accuracy_val)

        torch.save(training_log, os.path.join(args.save, 'training_log'))
        save_model(model, 'model', args.save)

        if epoch % 1 == 0:
            save_model(model, 'model', args.save)
        
        time_b = datetime.datetime.now()
        print('ETA:{}s/{}s'.format((time_b - time_a).seconds, (time_b - time_a).seconds * (args.epoch - epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=2)
    parser.add_argument('-sv', '--save', default='./model/proto')
    args = parser.parse_args()
    print(vars(args))
    train(args)

