import numpy as np
import keras.backend as K

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def predictResponse_into_nparray(response, output_tensor_name):
    dims = response.outputs[output_tensor_name].tensor_shape.dim
    shape = tuple(d.size for d in dims)
    print(shape)
    return np.reshape(response.outputs[output_tensor_name].float_val, shape)

def map_metric(ytrue, ypred,minoverlap, nr_of_classes):

    ytrue = K.eval(ytrue)
    ypred = K.eval(ypred)
    print((ypred))
    sum_AP = 0.0
    MINOVERLAP = minoverlap
    nr_classes = nr_of_classes
    gt_counter_per_class = np.zeros((1, 3))
    count_true_positives = np.zeros((1, 3))
    true_boxes = ytrue
    pred_boxes = ypred
    print(ytrue[0])
    print(np.array(ypred))

    for class_index in range(nr_classes):
        gt_counter_per_class[class_index] = np.sum(true_boxes[:, -1] == class_index)

        """
         Assign predictions to ground truth objects
        """
        nd = len(pred_boxes)
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, prediction in enumerate(pred_boxes):

            # assign prediction to ground truth object if any
            #   open ground-truth with that file_id

            ovmax = -1
            gt_match = -1
            # load prediction bounding-box
            bb = [float(x) for x in pred_boxes[1:5]]
            for obj in true_boxes:
                # look for a class_name match
                if obj[4] == class_index:
                    bbgt = [float(x) for x in obj[:4]]
                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # assign prediction as true positive or false positive

            # set minimum overlap
            min_overlap = MINOVERLAP

            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[idx] += 1

                else:
                    # false positive (multiple detection)
                    fp[idx] = 1

            else:
                # false positive
                fp[idx] = 1

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        # print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_index]
        # print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        # print(prec)

        ap, mrec, mprec = voc_ap(rec, prec)
        sum_AP += ap

    mAP = sum_AP / nr_classes

    return mAP