import numpy as np
import torch


class Losses:

    def pixel_accuracy(self, output, notHot_mask):  # accuracy= (TP+TN)/(TP+TN+FP+FN) =correct/total
        with torch.no_grad():
            output = torch.argmax(output, dim=1)  # when class 1 has highest propability, insert 1 etc.
            correct = torch.eq(output, notHot_mask).int()  # equal?
            accuracy = float(correct.sum()) / float(correct.numel())
        return accuracy

    # def hausdorfdistance(self,pred_mask, mask):
    #     num_of_labels = pred_mask.shape[1]
    #     #XA = pred_mask.permute(1, 0, 2, 3, 4) #[9,144,144,22,4]
    #     XA = torch.argmax(pred_mask, dim=1)
    #     XA = F.one_hot(XA, num_of_labels)
    #     XA = XA.permute(0,4,1,2,3)
    #     XB = mask.permute(0,4,1,2,3)
    #     result = monai.metrics.compute_hausdorff_distance(XA, XB,include_background=True, distance_metric='euclidean')
    #
    #     Xmin = 0
    #     dimx = mask.shape[1]
    #     dimy = mask.shape[2]
    #     dimz = mask.shape[3]
    #     Xmax = math.sqrt((math.pow(dimx, 2) + math.pow(dimy, 2) + math.pow(dimz, 2)))
    #     result = torch.nan_to_num(result, Xmax)
    #     result[result == float("Inf")] = Xmax
    #     result[result > Xmax] = Xmax
    #     normalized_hd = ((result) - Xmin) / (Xmax - Xmin)
    #
    #     meanforallbatchesandclasses=torch.mean(normalized_hd)
    #
    #     return meanforallbatchesandclasses

    # def averagehausdorfdistance(self,pred_mask, mask):
    #     num_of_labels = pred_mask.shape[1]
    #     #XA = pred_mask.permute(1, 0, 2, 3, 4) #[9,144,144,22,4]
    #     XA = torch.argmax(pred_mask, dim=1)
    #     XA = F.one_hot(XA, num_of_labels)
    #     XA = XA.permute(0,4,1,2,3)
    #     XB = mask.permute(0,4,1,2,3)
    #     result = monai.metrics.compute_average_surface_distance(XA, XB,include_background=True, distance_metric='euclidean')
    #     Xmin=0
    #     dimx=mask.shape[1]
    #     dimy = mask.shape[2]
    #     dimz = mask.shape[3]
    #     Xmax=math.sqrt((math.pow(dimx,2)+math.pow(dimy,2)+math.pow(dimz,2)))
    #     result=torch.nan_to_num(result,Xmax)
    #     result[result==float("Inf")]=Xmax
    #     result[result > Xmax] = Xmax
    #     normalized_ahd=((result)-Xmin)/(Xmax-Xmin)
    #     #nanboolean=torch.isnan(normalized_ahd).any()
    #     #infboolean=torch.isinf(normalized_ahd).any()
    #
    #
    #     return torch.mean(normalized_ahd)

    def hausdorff_distance(self, pred_mask, mask):
        # indicesleftlabel = np.where(leftlabel == 1)
        XA = pred_mask.permute(1, 2, 3, 4, 0)  # [9,144,144,22,4]
        XB = mask.permute(4, 1, 2, 3, 0)  # [9,144,144,22,4]

        XA = torch.argmax(XA, dim=0)
        XB = torch.argmax(XB, dim=0)

        nA = XA.shape[0]
        nB = XB.shape[0]
        cmax = 0.
        for batch in range(XA[3]):

            # for label in range(XA[3]):
            for i in range(nA):  # alle in der x achse
                cmin = np.inf
                for j in range(nB):
                    d = self.euclideandistance(XA[i, :], XB[j, :])
                    if d < cmin:
                        cmin = d
                    if cmin < cmax:
                        break
                if cmin > cmax and np.inf > cmin:
                    cmax = cmin

            for j in range(nB):
                cmin = np.inf
                for i in range(nA):
                    d = self.euclideandistance(XA[i, :], XB[j, :])
                    if d < cmin:
                        cmin = d
                    if cmin < cmax:
                        break
                if cmin > cmax and np.inf > cmin:
                    cmax = cmin
        return cmax

    def mIoU(self, pred_mask, notHot_mask, smooth=1e-10, n_classes=9):
        with torch.no_grad():

            # pred_mask = F.softmax(pred_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1)
            pred_mask = pred_mask.contiguous().view(-1)
            # mask = torch.argmax(mask, dim=1)

            notHot_mask = notHot_mask.contiguous().view(-1)

            iou_per_class = []
            for clas in range(1, n_classes):  # loop per pixel class
                true_class = pred_mask == clas  # boolean with true, where clas was predicted
                true_label = notHot_mask == clas

                if true_label.long().sum().item() == 0:  # no exist label in this loop
                    iou_per_class.append(np.nan)
                else:

                    intersect = torch.logical_and(true_class, true_label).sum().float().item()
                    union = torch.logical_or(true_class, true_label).sum().float().item()

                    iou = (intersect + smooth) / (union + smooth)
                    iou_per_class.append(iou)
            return np.nanmean(iou_per_class)

    def tversky(self, pred_mask, mask, smooth=1, alpha=0.3):
        beta = 1 - alpha

        pred_mask = pred_mask.permute(1, 2, 3, 4, 0)
        mask = torch.permute(mask, (4, 1, 2, 3, 0))
        mask = torch.flatten(mask)

        pred_mask = torch.flatten(pred_mask)  # put all in 1d array
        TP = (mask * pred_mask).sum()  # calculate true positive
        FN = (mask * (1 - pred_mask)).sum()  # calculate false negative
        FP = ((1 - mask) * pred_mask).sum()  # calculate false positive

        Tversky = (TP + smooth) / (TP + beta * FN + alpha * FP + smooth)
        return 1 - Tversky

    def focal_tversky(self, pred_mask, mask, gamma=4 / 3):
        focaltver = torch.pow(self.tversky(pred_mask, mask), 1 / gamma)
        return focaltver

    def categorical_cross_entropy(self, pred_mask, mask):
        pred_mask = pred_mask.permute(1, 2, 3, 4, 0)
        mask = torch.permute(mask, (4, 1, 2, 3, 0))
        # pred_mask = F.softmax(pred_mask, dim=0)
        crossentr = torch.mean(torch.sum(torch.log(pred_mask) * mask, dim=0))
        return - crossentr

    def focal_loss(self, pred_mask, mask, alpha=0.25, gamma=2):
        crossentr = self.categorical_cross_entropy(pred_mask, mask)
        focalloss = torch.mean(alpha * torch.pow((1 - pred_mask), gamma) * crossentr)
        return focalloss

    def unified_focal_loss(self, pred_mask, mask, weight=0.5):
        focaltverskyloss = self.focal_tversky(pred_mask, mask)
        focalloss = self.focal_loss(pred_mask, mask)
        unifiedfocal = weight * focaltverskyloss + (1 - weight) * focalloss

        return unifiedfocal

    def euclideandistance(self, array_x, array_y):
        n = array_x.shape[0]
        m = array_x.shape[1]
        ret = 0.
        for i in range(n):
            for j in range(m):
                ret += (array_x[i, j] - array_y[i, j]) ** 2  # absolutwert durch **2 sqrt
        return torch.sqrt(ret)
