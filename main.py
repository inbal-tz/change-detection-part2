
import random
import time
import numpy as np
import torch
import os
from sklearn.metrics import precision_recall_curve
import math
import torch.nn as nn
from iou_cc import calc_iou

import cv2 #changedByAlex
#Importing library to do image related operations
from PIL import Image, ImageOps


#Importing the important functionalities of Pytorch such as the dataloader, Variable, transform's 
#and optimizer related functions.

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor, ToPILImage
from erfnet import Net

# Importing the dataset class for VOC12 and cityscapes
from dataset import cityscapes
from dataset import idd_lite

import sys

#Importing the Relabel, ToLabel and Colorize class from transform.py file
from transform import Relabel, ToLabel, Colorize
import matplotlib
from matplotlib import pyplot as plt


import importlib
from iouEval import iouEval, getColorEntry #importing iouEval class from the iouEval.py file
from shutil import copyfile


# ### A few global parameters ###


NUM_CHANNELS = 3 #RGB Images
NUM_CLASSES = 2 #IDD Lite has 8 labels or Level1 hierarchy of labels
USE_CUDA = torch.cuda.is_available()
IMAGE_HEIGHT = 64
OUTPUT_DIR = r'D:\Users Data\inbal.tlgip\Project\output_images'
DATA_ROOT = r'D:\Users Data\inbal.tlgip\Desktop\part b'
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_EPOCHS = 80
ENCODER_ONLY = False
device = torch.device("cuda")
# device = torch.device("cpu")
# device = 'cuda'
color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()


IOUTRAIN = False
IOUVAL = True


#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=160):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, input1, target):
        # Resizing data to required size
        # input =  Resize((self.height,320), Image.BILINEAR)(input) #askalex
        # input1 =  Resize((self.height,320), Image.BILINEAR)(input1) #ChangedByUs
        # target = Resize((self.height,320), Image.NEAREST)(target)

        if(self.augment):
            # Random horizontal flip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                input1 = input1.transpose(Image.FLIP_LEFT_RIGHT) #ChangedByUs
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            #Random translation 0-2 pixels (fill rest with padding)
            transX = random.randint(0, 2)
            transY = random.randint(0, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            input1 = ImageOps.expand(input1, border=(transX,transY,0,0), fill=0) #ChangedByUs
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 7  #askalex- change 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            input1 = input1.crop((0, 0, input1.size[0]-transX, input1.size[1]-transY)) #ChangedByUs
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

        input = ToTensor()(input)
        input1 = ToTensor()(input1) #ChangedByUs

        target = ToLabel()(target)

        #target = Relabel(255, 7)(target)
        return input, input1, target #ChangedByUs


# ### Loading Data
#
# We'll follow pytorch recommended semantics, and use a dataloader to load the data.


def calc_TPFPTNFN_on_test_set(model, loader_test):
    print('printing preformance on the test set: ', end='')
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for step, (images, images1, labels, filename) in enumerate(loader_test):
        inputs = images.to(device)
        inputs1 = images1.to(device)  # ChangedByUs
        targets = labels.to(device)
        targets[targets < 128] = 0  # ChangedByUs
        targets[targets >= 128] = 1  # ChangedByUs
        output, _ = model([inputs.to(device), inputs1.to(device)], only_encode=ENCODER_ONLY)
        output = output[0]
        target = torch.LongTensor([target.cpu().numpy().flatten()[0] for target in targets])[0]
        if target == 0 and output[0] > output[1]:
            TN += 1
        if target == 0 and output[0] < output[1]:
            FP += 1
        if target == 1 and output[0] > output[1]:
            FN += 1
        if target == 1 and output[0] < output[1]:
            TP += 1
    print('FP: ', FP, 'FN: ', FN, 'TP: ', TP, 'TN: ', TN)
    if TP+FN > 0 and TP+FP > 0:
        print('recall= ', TP/(TP+FN), 'precision= ', TP/(TP+FP))


def main():
    best_acc = 0

    co_transform = MyCoTransform(ENCODER_ONLY, augment=True, height=IMAGE_HEIGHT)
    co_transform_val = MyCoTransform(ENCODER_ONLY, augment=False, height=IMAGE_HEIGHT)

    #train data
    dataset_train = idd_lite(DATA_ROOT, co_transform, 'train')
    print("length of training set: ",len(dataset_train))
    #test data
    dataset_test = idd_lite(DATA_ROOT, co_transform, 'test')
    print("length of validation set: ",len(dataset_test))

    # NOTE: PLEASE DON'T CHANGE batch_size and num_workers here. We have limited resources.
    loader_train = DataLoader(dataset_train, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(dataset_test, num_workers=NUM_WORKERS, batch_size=1, shuffle=True)
    dataiter = iter(loader_test)
    seven_test_images = []
    for i in range(7):
        res = dataiter.next()
        test_image_A = res[0]
        test_image_B = res[1]
        test_image_labels = res[2]
        seven_test_images.append((test_image_A.to(device), test_image_B.to(device)))
        cv2.imwrite(os.path.join(OUTPUT_DIR, str(i), 'A.tiff'),
                    np.rollaxis((test_image_A[0, :, :, :].squeeze().cpu().numpy() * 255).astype('uint8'), 0, 3))
        cv2.imwrite(os.path.join(OUTPUT_DIR, str(i), 'B.tiff'),
                    np.rollaxis((test_image_B[0, :, :, :].squeeze().cpu().numpy() * 255).astype('uint8'), 0, 3))
        cv2.imwrite(os.path.join(OUTPUT_DIR, str(i), 'label.tiff'),
                    (test_image_labels[0, :, :, :].squeeze().cpu().numpy()).astype('uint8'))

    # ## Cross Entropy  Loss ##
    # Negative Log Loss   |Plot of -log(x) vs x
    # - | -
    # ![alt](img/nll.png) | ![alt](img/nll-log.png)
    #
    # The negative log-likelihood becomes unhappy at smaller values, where it can reach infinite unhappiness (that’s too sad), and becomes less unhappy at larger values. Because we are summing the loss function to all the correct classes, what’s actually happening is that whenever the network assigns high confidence at the correct class, the unhappiness is low, but when the network assigns low confidence at the correct class, the unhappiness is high.

    criterion = torch.nn.CrossEntropyLoss()

    # ## Model ##
    model_file = importlib.import_module('erfnet')
    model = model_file.Net(NUM_CLASSES).to(device)

    # ### Optimizer ###
    # We use adam optimizer. It can be replaced with SGD and other optimizers
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    start_epoch = 1

    # ### Training Procedure ###
    softmax = torch.nn.Softmax(dim=1)

    steps_loss = 50
    my_start_time = time.time()
    for epoch in range(start_epoch, NUM_EPOCHS+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        epoch_loss = []
        time_train = []

        doIouTrain = IOUTRAIN
        doIouVal = IOUVAL

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        model.train()
        for step, (images, images1, labels, filename) in enumerate(loader_train):  # ChangedByUs
            start_time = time.time()
            # s = torch.sum(model.decoder.finalLayer.weight.data)  # show weights of final layer
            # inputs = [images.to(device), images1.to(device)] #ChangedByUs
            inputs = images.to(device)
            inputs1 = images1.to(device)  # ChangedByUs
            targets = labels.to(device)
            targets_orig = targets.clone()
            targets[targets_orig >= 128] = 1  # ChangedByUs 1=white. there is a change
            targets[targets_orig < 128] = 0  # ChangedByUs 0=black. no change
            outputs, _ = model([inputs, inputs1], only_encode=ENCODER_ONLY)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            # targets = max(targets)
            targets = torch.LongTensor([target.cpu().numpy().flatten()[0] for target in targets])
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if doIouTrain:
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            # print statistics
            if steps_loss > 0 and step % steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print('loss: {average:',average,'} (epoch: {',epoch,'}, step: {', step, '})', "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / BATCH_SIZE))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")

        # print FP FN TP TN
        calc_TPFPTNFN_on_test_set(model, loader_test)
        # print("results of 7 test images")
        # for image in seven_test_images:
        #     inputs = image[0].to(device)
        #     inputs1 = image[1].to(device)  # ChangedByUs
        #     output, _ = model([inputs.to(device), inputs1.to(device)], only_encode=ENCODER_ONLY)
        #     output = output[0]
        #     print(output)  # correct output for 12/2/2020 1001101
    my_end_time = time.time()
    print(my_end_time - my_start_time)

    print('loss: {average:', average, '} (epoch: {',epoch,'}, step: {',step,'})', "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / BATCH_SIZE))
    torch.save(model.state_dict(), r'C:\Users\inbal.tlgip\modelsave_cropped.pt')

    # # ### Validation ###
    # #Validate on val images after each epoch of training
    # print("----- VALIDATING - EPOCH", epoch, "-----")
    # model.eval()
    # epoch_loss_val = []
    # time_val = []
    #
    # if (doIouVal):
    #     iouEvalVal = iouEval(NUM_CLASSES)
    #
    # for step, (images, labels) in enumerate(loader_val):
    #     start_time = time.time()
    #
    #     inputs = images.to(device)
    #     targets = labels.to(device)
    #
    #     with torch.no_grad():
    #         outputs = model(inputs, only_encode=ENCODER_ONLY)
    #         #outputs = model(inputs)
    #     loss = criterion(outputs, targets[:, 0])
    #     epoch_loss_val.append(loss.item())
    #     time_val.append(time.time() - start_time)
    #
    #
    #     #Add batch to calculate TP, FP and FN for iou estimation
    #     if (doIouVal):
    #         #start_time_iou = time.time()
    #         iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
    #         #print ("Time to add confusion matrix: ", time.time() - start_time_iou)
    #
    #     if steps_loss > 0 and step % steps_loss == 0:
    #         average = sum(epoch_loss_val) / len(epoch_loss_val)
    #         print('VAL loss: {average:',average,'} (epoch: {',epoch,'}, step: {',step,'})',
    #                 "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / BATCH_SIZE))
    #
    #
    # average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
    #
    # iouVal = 0
    # if (doIouVal):
    #
    #     iouVal, iou_classes = iouEvalVal.getIoU()
    #     print(iou_classes)
    #     iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    #     print ("EPOCH IoU on VAL set: ", iouStr, "%")

    #
    #  ### Visualizing the Output###

    # Qualitative Analysis


    ##################### calc iou on test data #####################
    dataset_test = idd_lite(DATA_ROOT, co_transform_val, 'test')
    loader_test = DataLoader(dataset_test, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)
    # dataiter = iter(loader_test)
    # (val_image_A, val_image_B, val_image_labels) = dataiter.next()
    for step, (images, images1, labels, filename) in enumerate(loader_test):

        outputs_val, _ = model([images.to(device), images1.to(device)], only_encode=ENCODER_ONLY)
        outputs_val = softmax(outputs_val)
        # cv2.imwrite(r'D:\Users Data\inbal.tlgip\Project\output_images\test_output/'+str(step)+'.tiff',
        #             (((outputs_val[0, 1, :, :] > 0.5) * 255).squeeze().cpu().numpy()).astype('uint8'))



if __name__ == '__main__':
    # main()
    # test model- load model from file
    softmax = torch.nn.Softmax(dim=1)
    model_file = importlib.import_module('erfnet')
    model = model_file.Net(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(r'C:\Users\inbal.tlgip\modelsave_cropped_80epochs_recall75_precission61_training80000_batch16.pt'))
    model.to(device)
    model.eval()
    # from torchsummary import summary
    # summary(model, (2,3,64,64))
    # print(model)
    co_transform_val = MyCoTransform(ENCODER_ONLY, augment=False, height=IMAGE_HEIGHT) #askAlex why we dont augment in val?
    #load test data
    dataset_test = idd_lite(DATA_ROOT, co_transform_val, 'test')
    loader_test = DataLoader(dataset_test, num_workers=NUM_WORKERS, batch_size=1, shuffle=True)
    # calc_TPFPTNFN_on_test_set(model, loader_test)
    recall_precission_curve_label=[]
    recall_precission_curve_pred =[]
    for step, (images, images1, labels, filename) in enumerate(loader_test):
        inputs = images.to(device)
        inputs1 = images1.to(device)  # ChangedByUs
        targets = labels.to(device)
        targets[targets < 128] = 0  # ChangedByUs
        targets[targets >= 128] = 1  # ChangedByUs
        try:
            output, GAP = model([inputs.to(device), inputs1.to(device)], only_encode=ENCODER_ONLY)
        except:
            continue
        output = output[0]
        target = torch.LongTensor([target.cpu().numpy().flatten()[0] for target in targets])[0]
        recall_precission_curve_label.append(target.item())
        recall_precission_curve_pred.append(torch.nn.Softmax(dim=0)(output)[1])
        #save 16 channels to gap folder
        # for i in range(16):
        #     cv2.imwrite("gap/gap"+str(i)+".tiff", (GAP[0, i, :, :].squeeze().cpu().detach().numpy()).astype('uint8'))
        # import shutil
        # shutil.copy(os.path.join(r'D:\Users Data\inbal.tlGIP\Desktop\part b\labels\testCropped', filename[0] + '.jpg'), "gap/label.tiff")
        # shutil.copy(os.path.join(r'D:\Users Data\inbal.tlGIP\Desktop\part b\images\test\A', filename[0] + '.jpg'), "gap/A.tiff")
        # shutil.copy(os.path.join(r'D:\Users Data\inbal.tlGIP\Desktop\part b\images\test\B', filename[0] + '.jpg'), "gap/B.tiff")
        weights = model.decoder.finalLayer.weight
        final_image = GAP[0, 0, :, :] * weights[1][0]
        for i in range(1, 16):
            final_image += GAP[0, i, :, :] * weights[1][i]
        cv2.imwrite("heat_map.tiff", final_image.squeeze().cpu().detach().numpy())
        fig , ax= plt.subplots()
        ax.imshow(final_image.squeeze().cpu().detach(), cmap = 'jet', )
        fig.savefig('gap/'+filename[0]+'.png')
        plt.close(fig)
        # print(filename)
        # print("target:", target, "output:", torch.nn.Softmax()(output))
    #     # print()
        x = 1
    precision, recall, thresholds = precision_recall_curve(recall_precission_curve_label, recall_precission_curve_pred)
    best_avg = 0
    best_threshold = 0
    best_precision=0
    best_recall=0

    for i in range(len(precision)):
        cur_avg = 2*(precision[i] * recall[i])/(precision[i] + recall[i])
        if cur_avg > best_avg:
            best_avg = cur_avg
            best_threshold = thresholds[i]
            best_precision = precision[i]
            best_recall = recall[i]
    print(best_avg, best_threshold, best_precision, best_recall)
    print(thresholds.tolist())
    # print( precision)
    # print( recall)
    # print( thresholds)
    plt.scatter(recall, precision)
    plt.show()



