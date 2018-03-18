import torch

def calculate_thresholded_precision(pred, target, thresholds):
    batch_size = pred.size()[0]
    thresholded_precisions = [calculate_IOU(pred, target, t / 100.0) for t in thresholds]
    avg_precision_per_sample = torch.cat(thresholded_precisions, 0).view(len(thresholded_precisions),
                                                                         batch_size).permute(1, 0).mean(1)
    avg_precision = avg_precision_per_sample.mean()
    return avg_precision

def calculate_IOU(pred, target, threshold):
    batch_size = pred.size()[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    tp = torch.sum(((pred_flat >= threshold) * (target_flat == 1)).float(), 1)
    fp = torch.sum(((pred_flat >= threshold) * (target_flat == 0)).float(), 1)
    fn = torch.sum(((pred_flat < threshold) * (target_flat == 1)).float(), 1)
    precision = tp.float()/(tp + fp + fn).float()
    return precision

def create_thresholded_mask(pred, thresholds):
    return {str(t / 100.0): (pred >= t/100.0).float().squeeze(1).data.numpy() for t in thresholds}

