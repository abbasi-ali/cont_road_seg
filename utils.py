import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.eye(self.num_classes).to(target.device)[target]
        target = target.permute(0, 3, 1, 2).float()

        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = torch.where(target == 1, 1 - pt, pt)
        focal_weight = self.alpha * torch.pow(focal_weight, self.gamma)

        bce_loss = -target * torch.log(pred + self.eps) - (1 - target) * torch.log(1 - pred + self.eps)
        focal_loss = focal_weight * bce_loss

        return torch.mean(focal_loss)
    
class FocalLossWeighted(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, eps=1e-6):
        super(FocalLossWeighted, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.eye(self.num_classes).to(target.device)[target]
        target = target.permute(0, 3, 1, 2).float()

        pt = torch.where(target == 1, pred, 1 - pred)

        c_weights = torch.zeros(self.num_classes).to(target.device)
        for c in range(self.num_classes):
            c_weights[c] = torch.sum(target[:, c, :, :]).item()

        c_weights = c_weights / torch.sum(c_weights)

        bce_loss = -target * torch.log(pred + self.eps) - (1 - target) * torch.log(1 - pred + self.eps)
        # print(c_weights.shape, bce_loss.shape)
        focal_loss = bce_loss * c_weights.reshape(1, self.num_classes, 1, 1)

        return torch.sum(focal_loss)
    

def calculate_iou(pred, target, num_classes, weighted = True):
    hist = torch.zeros((num_classes, num_classes), device=pred.device)
    
    # Confusion matrix
    for c in range(num_classes):
        mask_pred = pred == c
        for t in range(num_classes):
            mask_target = target == t
            hist[c, t] = torch.sum(mask_pred & mask_target).item()

    # Calculate IoU
    intersection = torch.diag(hist)
    union = hist.sum(dim=0) + hist.sum(dim=1) - intersection
    iou = intersection / (union + torch.finfo(float).eps)

    if weighted == False:
        return iou.mean().item() * 100
    else:
        # Calculate class weights
        class_weights = hist.sum(dim=0) / hist.sum()
        return (iou * class_weights).sum().item() * 100




# Example usage
# num_classes = 19  # Number of classes in your segmentation problem
# criterion = FocalLoss(num_classes=num_classes)
