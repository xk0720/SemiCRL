import numpy as np
import torch
import torch.nn.functional as F


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    target_softmax = target_logits
    # num_classes = input_logits.size()[1]
    return (input_softmax - target_softmax) ** 2 # (unlabeled_bs, num_classes, H, W, D)
    # return F.mse_loss(input_softmax, target_softmax, size_average=False)


def softmax_l2_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns pixel-level L2 loss (equivalent to 1 - cosine similarity)
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    # target_softmax = F.softmax(target_logits, dim=1)
    target_softmax = target_logits

    # (1 - cosine similarity)
    return 1 - F.cosine_similarity(input_softmax, target_softmax, dim=1) # (unlabeled_bs, H, W, D)


def compute_consist_loss_BCP(args, pred_stu, prob_tea, percent, target):
    """
    :param pred_stu: (unlabeled_bs, num_classes, H, W, D), logits
    :param prob_tea: (unlabeled_bs, num_classes, H, W, D), probabilities
    :param percent: int
    :param target: (unlabeled_bs, H, W, D), {0, 1, ... , num_classes}
    :return: consistency loss (pixel-wise cross entropy loss)
    """
    bs, num_class, h, w, d = pred_stu.shape

    with torch.no_grad(): # drop pixels with high entropy (high uncertainty)
        # compute the entropy (uncertainty)
        entropy = -torch.sum(prob_tea * torch.log(prob_tea + 1e-10), dim=1)  # (unlabeled_bs, H, W, D)
        thresh = np.percentile(
            entropy.detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() # (unlabeled_bs, H, W, D)
        target[thresh_mask] = 255
        weight = bs * h * w * d / torch.sum(target != 255)

    if args.pseudo_label_type == 'hard':
        loss = weight * F.cross_entropy(pred_stu, target, ignore_index=255)
    elif args.pseudo_label_type == 'soft':
        # choices: "mse", "l2", "ce"
        if args.consistency_loss_type == 'ce':
            loss_ = F.cross_entropy(pred_stu, prob_tea, reduction='none')  # (unlabeled_bs, H, W, D)
            loss = torch.sum(loss_ * thresh_mask) / (torch.sum(thresh_mask) + 1e-16)
        if args.consistency_loss_type == 'mse':
            loss_ = softmax_mse_loss(pred_stu, prob_tea)
            loss = torch.sum(loss_ * thresh_mask.unsqueeze(1)) / (args.num_classes * torch.sum(thresh_mask) + 1e-16)
        if args.consistency_loss_type == 'l2':
            loss_ = softmax_l2_loss(pred_stu, prob_tea)
            loss = torch.sum(loss_ * thresh_mask) / (torch.sum(thresh_mask) + 1e-16)
    else:
        raise ValueError('not supported')

    return loss


def compute_consist_loss(args, pred_stu, prob_tea, percent, target, wnet_weight=None):
    """
    :param pred_stu: (unlabeled_bs, num_classes, H, W, D), logits
    :param prob_tea: (unlabeled_bs, num_classes, H, W, D), probabilities
    :param percent: int
    :param target: (unlabeled_bs, H, W, D), {0, 1, ... , num_classes}
    :return: consistency loss (pixel-wise cross entropy loss)
    """
    bs, num_class, h, w, d = pred_stu.shape

    with torch.no_grad(): # drop pixels with high entropy (high uncertainty)
        # compute the entropy (uncertainty)
        entropy = -torch.sum(prob_tea * torch.log(prob_tea + 1e-10), dim=1)  # (unlabeled_bs, H, W, D)
        thresh = np.percentile(
            entropy.detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() # (unlabeled_bs, H, W, D)
        target[thresh_mask] = 255
        weight = bs * h * w * d / torch.sum(target != 255)

    if args.pseudo_label_type == 'hard':
        loss = weight * F.cross_entropy(pred_stu, target, ignore_index=255)
    elif args.pseudo_label_type == 'soft':
        # choices: "mse", "l2", "ce"
        if args.consistency_loss_type == 'ce':
            loss_ = F.cross_entropy(pred_stu, prob_tea, reduction='none')  # (unlabeled_bs, H, W, D)
            if wnet_weight is None:
                loss = torch.sum(loss_ * thresh_mask) / (torch.sum(thresh_mask) + 1e-16)
            else:
                loss = torch.sum(loss_ * thresh_mask.unsqueeze(1) * wnet_weight) / \
                       (torch.sum(thresh_mask.unsqueeze(1) * wnet_weight) + 1e-16)
        if args.consistency_loss_type == 'mse':
            loss_ = softmax_mse_loss(pred_stu, prob_tea)
            if wnet_weight is None:
                loss = torch.sum(loss_ * thresh_mask.unsqueeze(1)) / (args.num_classes * torch.sum(thresh_mask) + 1e-16)
            else:
                loss = torch.sum(loss_ * thresh_mask.unsqueeze(1) * wnet_weight) / \
                       (torch.sum(thresh_mask.unsqueeze(1) * wnet_weight) + 1e-16)
        if args.consistency_loss_type == 'l2':
            loss_ = softmax_l2_loss(pred_stu, prob_tea)
            if wnet_weight is None:
                loss = torch.sum(loss_ * thresh_mask) / (torch.sum(thresh_mask) + 1e-16)
            else:
                loss = torch.sum(loss_ * thresh_mask.unsqueeze(1) * wnet_weight) / \
                       (torch.sum(thresh_mask.unsqueeze(1) * wnet_weight) + 1e-16)
    else:
        raise ValueError('not supported')

    return loss
