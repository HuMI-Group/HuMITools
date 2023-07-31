import math
import os
import time

import numpy as np
import torch
import tqdm
from torchviz import make_dot
import torchvision
from torchview import draw_graph
from humi_pytorch.Losses import Losses


def fit(settings, device, epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0
    loss_class = Losses()
    lossfunction = getattr(loss_class, criterion)  # globals()[criterion]

    alreadyswitched = False
    #secondlossfunction = Losses.hausdorff_distance
    weight = 0.3
    model.to(device)
    fit_time = time.time()

    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        batches_train = 0
        batch = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            # training phase
            batch += 1
            # try:
            image_tiles, mask_tiles = data

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)

            # if (batch == 1 and (e % 2 == 0)) and save_mode == True:
            #    save_this_output(notHot_mask, e)
             # check if loss switched while training
            loss = lossfunction(output, mask)

            # testloss1=hausdorfdistance(output, mask)
            # testloss2 = averagehausdorfdistance(output, mask)
            # evaluation metrics
            # iou_score += mIoU(output, mask.unsqueeze(1))
            # accuracy += pixel_accuracy(output, mask.unsqueeze(1))
            # backward
            if not math.isnan(loss.item()) and not math.isinf(loss.item()):
                running_loss += loss.item()
                batches_train += 1

            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()
        # except:
        #    pass

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            batches_val = 0
            # validation loop
            with torch.no_grad():  # no traning with validationdata
                for i, data in enumerate(tqdm.tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    notHot_mask = mask.argmax(-1)

                    # evaluation metrics
                    val_iou_score += loss_class.mIoU(output, notHot_mask)
                    test_accuracy += loss_class.pixel_accuracy(output, notHot_mask)
                    # loss
                    loss = lossfunction(output, mask)


                    if not math.isnan(loss.item()) and not math.isinf(loss.item()):
                        test_loss += loss.item()
                        batches_val += 1

            # calculatio mean for each batch
            try:
                train_losses.append(running_loss / batches_train)
                test_losses.append(test_loss / batches_val)
            except:
                pass

            if min_loss > (test_loss / batches_val):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / batches_val)))
                min_loss = (test_loss / batches_val)
                decrease += 1
                if decrease % 5 == 0:
                    print('saving model...')
                    torch.save(model, settings.output_folder + '/' + settings.model + '.pt')

            if (test_loss / batches_val) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')

            val_iou.append(val_iou_score / batches_val)
            train_iou.append(iou_score / batches_train)
            train_acc.append(accuracy / batches_train)
            val_acc.append(test_accuracy / batches_val)
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / batches_train),
                  "Val Loss: {:.3f}..".format(test_loss / batches_val),
                  "Train mIoU:{:.3f}..".format(iou_score / batches_train),
                  "Val mIoU: {:.3f}..".format(val_iou_score / batches_val),
                  "Train Acc:{:.3f}..".format(accuracy / batches_train),
                  "Val Acc:{:.3f}..".format(test_accuracy / batches_val),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    # save_as_onnx(model,settings)
    return history


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



