import numpy as np
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output
from model.UNetFormer import UNetFormer as UNetFormer
from model.FTUNetFormer import ft_unetformer as FTUNetFormer
from model.ABCNet import ABCNet
from model.CMTFNet.CMTFNet import CMTFNet


def maybe_clear_output():
    # Default off in terminal logs; enable with SAM_RS_CLEAR_OUTPUT=1 when using notebooks.
    if os.environ.get("SAM_RS_CLEAR_OUTPUT", "0") == "1":
        clear_output(wait=True)

DATASET = os.environ.get("SAM_RS_DATASET", 'Vaihingen')
# DATASET = 'Urban'

if DATASET == 'Vaihingen':
    from utils import *
elif DATASET == 'Urban':
     from utils_loveda import *

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

if MODEL == 'UNetformer':
    net = UNetFormer(num_classes=N_CLASSES).cuda()
elif MODEL == 'FTUNetformer':
    net = FTUNetFormer(num_=N_CLASSES).cuda()
elif MODEL == 'ABCNet':
    net = ABCNet(num_classes=N_CLASSES).cuda()
elif MODEL == 'CMTFNet':
    net = CMTFNet(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

# Load the datasets
print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.01
LBABDA_BDY = 0.1
LBABDA_OBJ = 1.0
print("LBABDA_BDY: ", LBABDA_BDY)
print("LBABDA_OBJ: ", LBABDA_OBJ)
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def metrics_from_cm(cm, label_values=LABELS):
    print("Confusion matrix :")
    print(cm)
    print("---")

    total = np.sum(cm)
    accuracy = 100 * np.trace(cm) / total if total > 0 else 0.0
    print("{} pixels processed".format(total))
    print("Total accuracy : {}".format(accuracy))
    print("---")

    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        denom = (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        F1Score[i] = 2.0 * cm[i, i] / denom if denom > 0 else np.nan

    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:])))
    print("---")

    pa = np.trace(cm) / float(total) if total > 0 else 0.0
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total) if total > 0 else 0.0
    kappa = (pa - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    print("Kappa: %.4f" % (kappa))

    denom = (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    MIoU = np.divide(np.diag(cm), denom, out=np.zeros_like(np.diag(cm), dtype=float), where=denom != 0)
    print(MIoU)
    MIoU = np.nanmean(MIoU[:])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")
    return MIoU

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    data_folder = globals().get('DATA_FOLDER_TEST', DATA_FOLDER)
    label_folder = globals().get('LABEL_FOLDER_TEST', LABEL_FOLDER)
    eroded_folder = globals().get('ERODED_FOLDER_TEST', ERODED_FOLDER)

    test_images = (1 / 255 * np.asarray(io.imread(data_folder.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(label_folder.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Urban':
        eroded_labels = ((np.asarray(io.imread(eroded_folder.format(id)), dtype='int64') - 1) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(eroded_folder.format(id))) for id in test_ids)
    all_preds = [] if all else None
    all_gts = [] if all else None
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) if not all else None
    # Switch the network to inference mode
    with torch.no_grad():
        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).cuda()

                # Do the inference
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            if all:
                all_preds.append(pred)
                all_gts.append(gt_e)
            else:
                cm += confusion_matrix(gt_e.ravel(), pred.ravel(), labels=list(range(N_CLASSES)))
            maybe_clear_output()

    if all:
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
    else:
        accuracy = metrics_from_cm(cm)
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.30
    criterionb = BoundaryLoss()
    criteriono = ObjectLoss()
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, boundary, object, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss_ce = loss_calc(output, target, weights)
            loss_boundary = criterionb(output, boundary)
            loss_object = criteriono(output, object)

            if LOSS == 'SEG':
                loss = loss_ce
            elif LOSS == 'SEG+BDY':
                loss = loss_ce + loss_boundary * LBABDA_BDY
            elif LOSS == 'SEG+OBJ':
                loss = loss_ce + loss_object * LBABDA_OBJ
            elif LOSS == 'SEG+BDY+OBJ':
                loss = loss_ce + loss_boundary * LBABDA_BDY + loss_object * LBABDA_OBJ
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            if iter_ == 0:
                mean_losses[iter_] = losses[iter_]
            else:
                mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                maybe_clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_ce: {:.6f}\tLoss_boundary: {:.6f}\tLoss_object: {:.6f}\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss_ce.data, loss_boundary.data, loss_object.data, loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            print('Validation (epoch {}/{}): mIoU={:.4f}, best={:.4f}'.format(e, epochs, MIoU, MIoU_best))
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    os.makedirs('./resultsv', exist_ok=True)
                    torch.save(net.state_dict(), './resultsv/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                elif DATASET == 'Urban':
                    os.makedirs('./resultsu', exist_ok=True)
                    torch.save(net.state_dict(), './resultsu/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU

if MODE == 'Train':
    save_epoch = int(os.environ.get("SAM_RS_SAVE_EPOCH", "5"))
    train(net, optimizer, 50, scheduler, save_epoch=save_epoch)
elif MODE == 'Test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load('./resultsv/YOUR_MODEL')) # sam
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./resultsv/inference_'+MODEL+'_tile_{}.png'.format(id_), img)

    elif DATASET == 'Urban':
        net.load_state_dict(torch.load('./resultsu/YOUR_MODEL')) # sam
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
        print("MIoU: ", MIoU)
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            # plt.imshow(img) and plt.show()
            io.imsave('./resultsu/inference_'+MODEL+'_tile_{}.png'.format(id_), img)
