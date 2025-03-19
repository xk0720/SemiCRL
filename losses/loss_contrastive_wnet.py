import torch
from torch.nn import functional as F


def compute_contra_memobank_loss(
        args,
        proj_stu,
        proj_tea,
        label_l,
        label_u,
        prob_l,
        prob_u,
        low_mask,
        high_mask,
        memobank,
        queue_prtlis,
        queue_size,
        prototype_vectors, # prototype (centroid for each class)
        prototype_vectors_num, # the number of vectors for each prototype (centroid)
        iteration=0,
        logging=None,
        wnet_weight=None, # (unlabeled, h, w, d, 1)
        update_mode=True,
):

    # Problem: adjustable for different datasets containing different 'num_classes'
    # low_rank, high_rank = args.low_rank, args.high_rank
    # num_feat = proj_stu.shape[1] # 256 as the default

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask # (bs, num_classes, h, w, d), for anchors selecting
    # high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask # (bs, num_classes, h, w, d), for negatives selecting

    proj_stu = proj_stu.permute(0, 2, 3, 4, 1) # (bs, proj_dim, h, w, d) --> (bs, h, w, d, proj_dim)
    proj_tea = proj_tea.permute(0, 2, 3, 4, 1) # (bs, proj_dim, h, w, d) --> (bs, h, w, d, proj_dim)

    seg_weight_low_entropy_list = [] # candidate anchor pixels (weights)
    seg_feat_low_entropy_list = []  # candidate anchor pixels (features)
    # seg_num_list = []  # the number of low_valid pixels (features) in each class
    seg_proto_list = []  # the prototype (centroid) of each class

    valid_classes_contras = [] # valid_classes for compute the contrastive loss
    valid_classes_proto = [] # valid_classes for update the prototypes
    # valid_classes = [] # record the valid classes

    _, prob_indices_l = torch.sort(prob_l, 1, True) # return the indices
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 4, 1)  # (labeled_bs, h, w, d, num_classes)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 4, 1)  # (unlabeled_bs, h, w, d, num_classes)

    prob_all = torch.cat((prob_l, prob_u), dim=0)  # (bs, num_classes, h, w, d)
    label_all = torch.cat((label_l, label_u), dim=0) # (bs, num_classes, h, w, d)

    for i in range(args.centroids_num):
        # select binary mask for i-th class, for anchors
        low_valid_pixel_seg = low_valid_pixel[:, i]
        # select binary mask for i-th class, for negatives
        # high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob_all[:, i, :, :, :]

        # reliable pixels for constructing anchors
        # Problem: current_class_threshold is adjusted according to 'num_classes'
        rep_mask_low_entropy = (prob_seg > args.current_class_threshold) * \
                               low_valid_pixel_seg.bool() # (bs, h, w, d)

        # unreliable pixels for constructing negative samples
        rep_mask_high_entropy = high_mask.squeeze(1).bool() # (bs, h, w, d)

        # mask of anchors for i-th class
        anchor_mask = rep_mask_low_entropy
        seg_feat_low_entropy_list.append(proj_stu[anchor_mask])
        seg_weight_low_entropy_list.append(wnet_weight[anchor_mask])

        # TODO: whether use low_valid_pixel_seg or rep_mask_low_entropy as the mask to sample positive samples
        # positive sample: centroids of the class (compute based on the feature from teacher's projection head)
        average_proto = torch.mean(proj_tea[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True) # (1, proj_dim)
        # if args.proto_weight_mode == "similarity": # obtain the normalized prototype
        #     average_proto = F.normalize(average_proto, p=2, dim=1)
        seg_proto_list.append(average_proto)

        # generate mask for labeled data
        # only suitable for binary segmentation, (labeld_bs, h, w, d)
        if args.dataset == 'LA_Heart' or args.dataset == 'Pancreas' or args.dataset == 'BraTS2019':
            class_mask_l = torch.sum(prob_indices_l[:, :, :, :, :args.high_rank].eq(i), dim=4).bool()
        # in the case of multi-class segmentation
        else:
            class_mask_l = torch.sum(prob_indices_l[:, :, :, :, :(args.high_rank-1)].eq(i), dim=4).bool()

        # generate mask for unlabeld data, (unlabeld_bs, h, w, d)
        class_mask_u = torch.sum(prob_indices_u[:, :, :, :, args.low_rank:args.high_rank].eq(i), dim=4).bool()

        # generate the mask for the whole data
        class_mask = torch.cat((class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0)  # (bs, h, w, d)

        # mask of negative samples for i-th class
        negative_mask = rep_mask_high_entropy * class_mask

        # negative samples for i-th class
        keys = proj_tea[negative_mask.bool()].detach()  # tensor: (nums, proj_dim)

        # the reliability scores for negative samples for i-th class
        scores = wnet_weight[negative_mask.bool()] # tensors: (nums, 1)

        # print('the size of key is: ', len(keys))
        # print('the number of valid voxels in labeled data is: ', torch.sum(class_mask_l * (label_l[:, i] == 0)))

        # update the memory bank
        if update_mode:
            dequeue_and_enqueue(
                keys=keys,
                scores=scores,
                queue=memobank[i],
                # memobank: list; queue: list with length 1; queue[0]: tensor with shape (_, proj_dim)
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )

        # valid_classes for compute the contrastive loss
        if rep_mask_low_entropy.sum() > args.num_queries:
            # seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes_contras.append(i)

        # valid_classes for update the prototypes
        if low_valid_pixel_seg.sum() > 0:
            # seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes_proto.append(i)

    if len(valid_classes_proto) < 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return prototype_vectors, prototype_vectors_num, torch.tensor(0.0) * proj_stu.sum()

    else:
        contras_loss = torch.tensor(0.0).cuda()
        # seg_prototype = torch.cat(seg_proto_list)  # shape: (valid_classes_num, proj_dim)

        # update the prototype for each valid class (centroid)
        if update_mode:
            prototype_vectors, prototype_vectors_num = update_prototype(
                args=args,
                feat=proj_tea,
                prob_all=prob_all,
                label_all=label_all,
                low_valid_pixel=low_valid_pixel,
                momentum_prototype=prototype_vectors,
                seg_proto_list=seg_proto_list,
                prototype_vectors_num=prototype_vectors_num,
                valid_classes=valid_classes_proto,
                iteration=iteration,
                name='moving_average',
                maximum_num=3000,  # the max number of vectors for each prototype (centroid)
                update_mode='whole',
            )

        for i in range(len(valid_classes_contras)):
            class_idx = valid_classes_contras[i]
            if memobank[class_idx][0].shape[0] > 0:
                # TODO: Note that we apply the random sampling, may cause some problems,
                #  or maybe try sampling method based on the confidence weight?
                # randomly choose the indices of anchors
                anchor_idx = torch.randint(len(seg_feat_low_entropy_list[class_idx]), size=(args.num_queries,))
                # sample the anchor features
                anchor_feat = seg_feat_low_entropy_list[class_idx][anchor_idx].clone().cuda() # (num_queries, proj_dim)
                # sample the anchor weights
                anchor_weight = seg_weight_low_entropy_list[class_idx][anchor_idx].clone().cuda() # (num_queries, 1)

                anchor_feat = anchor_weight * anchor_feat
            else:
                contras_loss = contras_loss + 0 * proj_stu.sum()
                logging.info('{}-class is invalid.'.format(class_idx))
                continue

            with torch.no_grad():
                negative_feat = memobank[class_idx][0].clone().cuda()
                negative_weight = memobank[class_idx][1].clone().cuda()

                negative_feat_idx = torch.randint(len(negative_feat),
                                                  size=(args.num_queries * args.num_negatives,))
                negative_feat = negative_feat[negative_feat_idx]
                negative_weight = negative_weight[negative_feat_idx]

                negative_feat = negative_weight * negative_feat
                # negative samples in contrastive loss; (num_queries, num_negatives, proj_dim)
                negative_feat = negative_feat.reshape(args.num_queries, args.num_negatives, args.proj_dim)

                # positive samples in contrastive loss; (num_queries, 1, proj_dim)
                positive_feat = prototype_vectors[i].clone().unsqueeze(0).unsqueeze(0).repeat(args.num_queries, 1, 1).cuda()

                # all samples against anchor samples
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)  # (num_queries, 1 + num_negatives, proj_dim)

            # compute the cosine similarity
            logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2) # (num_queries, 1 + num_negatives)

            # compute the contrastive loss
            contras_loss = contras_loss + \
                           F.cross_entropy(logits / args.temp, torch.zeros(args.num_queries).long().cuda())

            contras_loss = args.contrastive * contras_loss

        return prototype_vectors, prototype_vectors_num, contras_loss


def dequeue_and_enqueue(keys, scores, queue, queue_ptr, queue_size):
    """
    Purpose: update the queue for negative samples
    """
    batch_size = keys.shape[0]
    ptr = int(queue_ptr) # the pointer
    print('the size of keys is: {}'.format(batch_size))

    if keys.shape[0] > queue_size:
        indices = torch.randint(keys.shape[0], size=(queue_size,))
        queue[0] = torch.cat((queue[0], keys.detach().clone().cpu()[indices]), dim=0) # first enqueue
        queue[1] = torch.cat((queue[1], scores.detach().clone().cpu()[indices]), dim=0)
    else:
        queue[0] = torch.cat((queue[0], keys.detach().clone().cpu()), dim=0) # first enqueue
        queue[1] = torch.cat((queue[1], scores.detach().clone().cpu()), dim=0)

    # queue[0] = torch.cat((queue[0], keys.detach().clone().cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :] # then dequeue
        queue[1] = queue[1][-queue_size:, :]  # then dequeue
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr
    # return batch_size


def update_prototype(args, feat, prob_all, label_all, low_valid_pixel, momentum_prototype, prototype_vectors_num, seg_proto_list,
                     valid_classes, iteration, name='moving_average', use_scale_factor=False,
                     maximum_num=None, update_mode='whole'):

    if update_mode == 'whole':
        # if not (momentum_prototype == 0).all(): # continue updating
        for i in range(len(valid_classes)):
            # the class index
            class_idx = valid_classes[i]
            ema_decay = min(1 - 1 / (iteration + 1), 0.999)
            momentum_prototype[class_idx] = (1 - ema_decay) * seg_proto_list[class_idx][0] + \
                                            ema_decay * momentum_prototype[class_idx]
    elif update_mode == 'single':
        prob_all_logits = torch.max(prob_all, dim=1)[0]
        onehot_label2update = (prob_all_logits > args.current_class_threshold) * low_valid_pixel[:, 0]  # (bs, h, w, d)
        onehot_label = label_all * onehot_label2update  # (bs, num_classes, h, w, d)

        if use_scale_factor:
            scale_factor = F.adaptive_avg_pool3d(onehot_label, 1)  # (bs, centroids_num, 1, 1, 1)
        else:
            scale_factor = 1.0

        vectors = []
        ids = []
        for n in range(len(prob_all.size(0))): # n-th sample in the mini-batch
            for t in range(args.centroids_num):
                if torch.sum(onehot_label[n, t]) == 0:
                    continue
                mean_vector = F.adaptive_avg_pool3d(feat[n] * onehot_label[n][t], 1) / scale_factor[n][t] # (proj_dim, 1, 1, 1)
                vectors.append(mean_vector.squeeze())
                ids.append(t)

        for t in range(len(ids)):
            new_vector = vectors[t]
            centroids_idx = ids[t]

            if name == 'moving_average':
                ema_decay = min(1 - 1 / iteration + 1, 0.999)
                momentum_prototype[centroids_idx] = (1 - ema_decay) * new_vector + ema_decay * momentum_prototype[centroids_idx]
            elif name == 'mean':
                all_vectors = momentum_prototype[centroids_idx] * prototype_vectors_num[centroids_idx]
                prototype_vectors_num[centroids_idx] += 1
                momentum_prototype[centroids_idx] = (all_vectors + new_vector) / prototype_vectors_num[centroids_idx]

                if maximum_num is not None:
                    prototype_vectors_num[t] = min(prototype_vectors_num[centroids_idx], maximum_num)
            else:
                raise ValueError("name not supported")
    else:
        raise ValueError("mode not supported")

    return momentum_prototype, prototype_vectors_num
