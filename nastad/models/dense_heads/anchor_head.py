import math
import torch
import torch.nn as nn
from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule
from ..utils.bbox_tools import compute_delta, delta_to_pred
from ..utils.post_processing import batched_nms


@HEADS.register_module()
class AnchorHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        cls_prior_prob=0.01,
    ):
        super(AnchorHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob

        self.scales = prior_generator.scales
        self.strides = prior_generator.strides

        # anchor generator
        self.prior_generator = build_prior_generator(prior_generator)

        # build layers
        self._init_layers()

        # loss
        self.assigner = build_loss(loss.assigner)
        self.sampler = build_loss(loss.sampler)
        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    def _init_layers(self):
        self.rpn_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.rpn_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=dict(type="relu"),
                )
            )

        # regression
        self.rpn_reg = nn.Conv1d(self.feat_channels, len(self.scales) * 2, kernel_size=1)

        # classification (no sigmoid in layers)
        self.rpn_cls = nn.Conv1d(self.feat_channels, len(self.scales) * 1, kernel_size=1)

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            torch.nn.init.constant_(self.rpn_cls.bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred = []
        reg_pred = []

        for feat, mask in zip(feat_list, mask_list):
            for i in range(self.num_convs):
                feat, mask = self.rpn_convs[i](feat, mask)

            cls_pred.append(self.rpn_cls(feat) * mask.unsqueeze(1).float())  # todo
            reg_pred.append(self.rpn_reg(feat) * mask.unsqueeze(1).float())  # todo

        anchors = self.prior_generator(feat_list)  # List: [k,2] 0~1

        # loss
        losses = self.losses(cls_pred, reg_pred, mask_list, anchors, gt_segments, gt_labels)

        # get proposals
        proposals, _ = self._get_nms_proposal_list(anchors, mask_list, cls_pred, reg_pred)
        return losses, proposals

    def forward_test(self, feat_list, mask_list):
        cls_pred = []
        reg_pred = []

        for feat, mask in zip(feat_list, mask_list):
            for i in range(self.num_convs):
                feat, mask = self.rpn_convs[i](feat, mask)

            cls_pred.append(self.rpn_cls(feat) * mask.unsqueeze(1).float())
            reg_pred.append(self.rpn_reg(feat) * mask.unsqueeze(1).float())

        anchors = self.prior_generator(feat_list)  # List: [k,2] 0~1

        # get proposals
        proposals, scores = self._get_nms_proposal_list(anchors, mask_list, cls_pred, reg_pred)
        return proposals, scores

    def losses(self, cls_pred, reg_pred, mask_list, anchors, gt_segments, gt_labels):
        bs, num_scales = cls_pred[0].shape[:2]

        cls_pred = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1)  # [B,K]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)  # [B,K,2]
        masks = torch.cat(mask_list, dim=-1).unsqueeze(-1).repeat(1, 1, num_scales).reshape(bs, -1)  # [B,K]

        # get gt targets and positive negative mask
        gt_cls, gt_reg, pos_idxs_list, neg_idxs_list = self.prepare_targets(anchors, masks, gt_segments, gt_labels)

        num_pos = sum([len(pos_idxs) for pos_idxs in pos_idxs_list])
        num_neg = sum([len(neg_idxs) for neg_idxs in neg_idxs_list])

        # classification loss
        sampled_cls_pred = []
        for pred, mask, pos_idxs, neg_idxs in zip(cls_pred, masks, pos_idxs_list, neg_idxs_list):
            sampled_cls_pred.append(pred[mask][pos_idxs + neg_idxs])
        sampled_cls_pred = torch.cat(sampled_cls_pred, dim=0)

        loss_cls = self.cls_loss(sampled_cls_pred, gt_cls.float())
        loss_cls /= num_pos + num_neg

        # regression loss
        sampled_reg_pred = []
        for pred, mask, pos_idxs in zip(reg_pred, masks, pos_idxs_list):
            sampled_reg_pred.append(pred[mask][pos_idxs])
        sampled_reg_pred = torch.cat(sampled_reg_pred, dim=0)

        if num_pos == 0:  # not have positive sample
            # do not have positive samples in regression loss
            loss_reg = torch.Tensor([0]).sum().to(reg_pred.device)
        else:
            loss_reg = self.reg_loss(sampled_reg_pred, gt_reg)
            loss_reg /= num_pos

        losses = {"rpn_cls": loss_cls, "rpn_reg": loss_reg}
        return losses

    @torch.no_grad()
    def prepare_targets(self, anchors, masks, gt_segments, gt_labels):
        # prepare gts: assign the gt_segment to each anchor
        anchors = torch.cat(anchors, dim=0)  # [B,K,2]

        gt_cls_list, gt_reg_list, pos_idxs_list, neg_idxs_list = [], [], [], []
        for i, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
            if len(gt_segment) == 0:  # make a pseudo gt_segment
                gt_segment = torch.tensor([[0, 0]], dtype=torch.float32, device=anchors.device)
                gt_label = torch.zeros(self.num_classes, device=anchors.device).to(torch.int64)
            else:
                gt_label = torch.ones((gt_segment.shape[0], 1), device=anchors.device).to(torch.int64)  # binary label

            # assign GT for valid positions
            _, assigned_gt_idxs, assigned_labels = self.assigner.assign(
                anchors[masks[i]],
                gt_segment,
                gt_label,
            )

            # sample positive and negative anchors
            pos_idxs, neg_idxs = self.sampler.sample(assigned_gt_idxs)
            pos_idxs_list.append(pos_idxs)
            neg_idxs_list.append(neg_idxs)

            # classification target: pos_mask + neg_mask
            gt_cls = assigned_labels[pos_idxs + neg_idxs].squeeze(-1)
            gt_cls_list.append(gt_cls)

            # regression target: pos_mask
            gt_reg = compute_delta(anchors[masks[i]][pos_idxs], gt_segment[assigned_gt_idxs[pos_idxs] - 1])
            gt_reg_list.append(gt_reg)

        gt_cls_concat = torch.cat(gt_cls_list, dim=0)  #  [B*(pos+neg)]
        gt_reg_concat = torch.cat(gt_reg_list, dim=0)  #  [B*(pos),2]
        return gt_cls_concat, gt_reg_concat, pos_idxs_list, neg_idxs_list

    @torch.no_grad()
    def _get_nms_proposal_list(self, anchors, mask_list, cls_pred, reg_pred):
        bs = cls_pred[0].shape[0]
        device = cls_pred[0].device
        pre_nms_topk = 2000
        post_nms_topk = 1000
        nms_thresh = 0.7

        # for each feature map, apply delta_to_pred() and select top-k anchors before nms
        topk_proposals, topk_scores, topk_masks, level_ids = [], [], [], []
        batch_idx = torch.arange(bs, device=device)
        for l, (anchor_i, logits_i, reg_i, mask_i) in enumerate(zip(anchors, cls_pred, reg_pred, mask_list)):
            # 1. get valid anchors
            mask_i = mask_i.unsqueeze(-1).repeat(1, 1, len(self.scales)).flatten(1)  # [bs,T*len(scales)]

            # 2. apply delta_to_pred() to get proposals
            reg_i = reg_i.permute(0, 2, 1).reshape(bs, -1, 2)
            scores_i = logits_i.permute(0, 2, 1).reshape(bs, -1).sigmoid()  # [bs, T*len(scales)]
            proposals_i = delta_to_pred(anchor_i, reg_i)  # [bs, T*len(scales), 2]

            # 3. select top-k anchor for each level and each video
            num_proposals_i = min(proposals_i.shape[1], pre_nms_topk)
            topk_scores_i, topk_idx = scores_i.topk(num_proposals_i, dim=1)
            topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # [bs,topk,2]
            topk_masks_i = mask_i[batch_idx[:, None], topk_idx]  # [bs,topk]

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            topk_masks.append(topk_masks_i)
            level_ids.append(torch.full((num_proposals_i,), l, dtype=torch.int64, device=device))

        # concat all levels together
        topk_proposals = torch.cat(topk_proposals, dim=1)
        topk_scores = torch.cat(topk_scores, dim=1)
        topk_masks = torch.cat(topk_masks, dim=1)
        level_ids = torch.cat(level_ids, dim=0)  # we have recorded the level id

        # NMS on each level, and choose topk results.
        nms_proposals, nms_scores = [], []
        for i in range(bs):
            # select valid proposals
            valid = topk_masks[i]

            # NMS on each feature map
            new_proposals, new_scores, _ = batched_nms(
                topk_proposals[i][valid],
                topk_scores[i][valid],
                level_ids[valid],
                iou_threshold=nms_thresh,
                max_seg_num=post_nms_topk,
                use_soft_nms=False,
                multiclass=True,
            )

            nms_proposals.append(new_proposals.to(device))
            nms_scores.append(new_scores.to(device))
        return nms_proposals, nms_scores

    # @torch.no_grad()
    def _get_proposal_list(self, anchors, mask_list, cls_pred, reg_pred):
        bs = cls_pred[0].shape[0]

        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)  # [B,T*len(scales),2]
        cls_pred = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1)  # [B,T*len(scales)]

        anchors = torch.cat(anchors, dim=0).unsqueeze(0)  # [1,T*len(scales),2]
        proposals = delta_to_pred(anchors, reg_pred.detach())  # [B,K,2]
        masks = torch.cat(mask_list, dim=1).unsqueeze(-1).repeat(1, 1, len(self.scales))  # [B,T,len(scales)]

        new_proposals, new_scores = [], []
        for proposal, logits, mask in zip(proposals, cls_pred, masks):
            new_proposals.append(proposal[mask.view(-1)])
            new_scores.append(logits[mask.view(-1)].detach().sigmoid())
        return new_proposals, new_scores


class AnchorHeadIB(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        cls_prior_prob=0.01,
        spacial_loss = 'SpacialCELoss',
        temporal_loss = 'TemporalLoss',
    ):
        super(AnchorHeadIB, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob

        self.scales = prior_generator.scales
        self.strides = prior_generator.strides

        # anchor generator
        self.prior_generator = build_prior_generator(prior_generator)

        # build layers
        self._init_layers()

        # loss
        self.assigner = build_loss(loss.assigner)
        self.sampler = build_loss(loss.sampler)
        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)
        self.spacial_loss = build_loss(spacial_loss)
        self.temporal_loss = build_loss(temporal_loss)


    def _init_layers(self):
        self.rpn_convs = nn.ModuleList([])
        # self.temporal_net = nn.ModuleList([])

        for i in range(self.num_convs):
            self.rpn_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=dict(type="relu"),
                )
            )

        # regression
        self.rpn_reg = nn.Conv1d(self.feat_channels, len(self.scales) * 2, kernel_size=1)
        self.temporal_mu_net=nn.Conv1d(self.feat_channels, len(self.scales) * 2, kernel_size=1)
        self.temporal_logvar_net=nn.Conv1d(self.feat_channels, len(self.scales) * 2, kernel_size=1)
        self.logvar_act=nn.Tanh()

        # classification (no sigmoid in layers)
        self.rpn_cls = nn.Conv1d(self.feat_channels, len(self.scales) * 1, kernel_size=1)
        self.spacial_variational_net=nn.Conv1d(self.feat_channels, len(self.scales) * 2, kernel_size=1)

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            torch.nn.init.constant_(self.rpn_cls.bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred = []
        reg_pred = []
        temporal_mu = []
        temporal_logvar= []
        spacial_variational = []


        for feat, mask in zip(feat_list, mask_list):
            for i in range(self.num_convs):
                feat, mask = self.rpn_convs[i](feat, mask)

            cls_pred.append(self.rpn_cls(feat) * mask.unsqueeze(1).float())  # todo
            reg_pred.append(self.rpn_reg(feat) * mask.unsqueeze(1).float())  # todo

            temporal_mu.append(self.temporal_mu_net(feat) * mask.unsqueeze(1).float())  #
            temporal_logvar.append(self.act(self.temporal_mu_net(feat) * mask.unsqueeze(1).float()))  #
            spacial_variational.append(self.spacial_variational_net(feat) * mask.unsqueeze(1).float())

        anchors = self.prior_generator(feat_list)  # List: [k,2] 0~1

        # loss
        losses = self.losses(cls_pred, reg_pred,temporal_mu,temporal_logvar, spacial_variational, mask_list, anchors, gt_segments, gt_labels)

        # get proposals
        proposals, _ = self._get_nms_proposal_list(anchors, mask_list, cls_pred, reg_pred)
        return losses, proposals

    def forward_test(self, feat_list, mask_list):
        cls_pred = []
        reg_pred = []

        for feat, mask in zip(feat_list, mask_list):
            for i in range(self.num_convs):
                feat, mask = self.rpn_convs[i](feat, mask)

            cls_pred.append(self.rpn_cls(feat) * mask.unsqueeze(1).float())
            reg_pred.append(self.rpn_reg(feat) * mask.unsqueeze(1).float())

        anchors = self.prior_generator(feat_list)  # List: [k,2] 0~1

        # get proposals
        proposals, scores = self._get_nms_proposal_list(anchors, mask_list, cls_pred, reg_pred)
        return proposals, scores

    def losses(self, cls_pred, reg_pred,temporal_mu,temporal_logvar, spacial_variational, mask_list, anchors, gt_segments, gt_labels):
        bs, num_scales = cls_pred[0].shape[:2]

        cls_pred = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1)  # [B,K]
        temporal_mu = torch.cat(temporal_mu, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)
        temporal_logvar = torch.cat(temporal_logvar, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)  # [B,K,2]
        spacial_variational = torch.cat(spacial_variational, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)
        masks = torch.cat(mask_list, dim=-1).unsqueeze(-1).repeat(1, 1, num_scales).reshape(bs, -1)  # [B,K]

        # get gt targets and positive negative mask
        gt_cls, gt_reg, pos_idxs_list, neg_idxs_list = self.prepare_targets(anchors, masks, gt_segments, gt_labels)

        num_pos = sum([len(pos_idxs) for pos_idxs in pos_idxs_list])
        num_neg = sum([len(neg_idxs) for neg_idxs in neg_idxs_list])

        # classification loss
        sampled_cls_pred = []
        sampled_spacial_variational = []
        for pred, mask, pos_idxs, neg_idxs in zip(cls_pred, masks, pos_idxs_list, neg_idxs_list):
            sampled_cls_pred.append(pred[mask][pos_idxs + neg_idxs])
        sampled_cls_pred = torch.cat(sampled_cls_pred, dim=0)

        for pred, mask, pos_idxs, neg_idxs in zip(spacial_variational, masks, pos_idxs_list, neg_idxs_list):
            sampled_spacial_variational.append(pred[mask][pos_idxs + neg_idxs])
        sampled_spacial_variational = torch.cat(sampled_spacial_variational, dim=0)

        loss_cls = self.cls_loss(sampled_cls_pred, gt_cls.float())
        spacial_mi, loss_spacial = self.spacial_loss(sampled_spacial_variational, gt_cls.float())
        loss_cls /= num_pos + num_neg
        loss_spacial /= num_pos + num_neg

        # regression loss
        sampled_reg_pred = []
        for pred, mask, pos_idxs in zip(reg_pred, masks, pos_idxs_list):
            sampled_reg_pred.append(pred[mask][pos_idxs])
        sampled_reg_pred = torch.cat(sampled_reg_pred, dim=0)

        sampled_temporal_mu = []
        sampled_temporal_logvar = []
        for mu, logvar, mask, pos_idxs in zip(temporal_mu,temporal_logvar, masks, pos_idxs_list):
            sampled_temporal_mu.append(mu[mask][pos_idxs])
            sampled_temporal_logvar.append(logvar[mask][pos_idxs])
        sampled_reg_mu = torch.cat(sampled_temporal_mu, dim=0)
        sampled_reg_logvar = torch.cat(sampled_temporal_logvar, dim=0)

        if num_pos == 0:  # not have positive sample
            # do not have positive samples in classification loss
            loss_temporal = torch.Tensor([0]).sum().to(temporal_mu.device)
            temporal_mi = 0
        else:
            temporal_mi, loss_temporal = self.temporal_loss(sampled_reg_mu, gt_reg,sampled_reg_logvar )
            loss_temporal /= num_pos

        if num_pos == 0:  # not have positive sample
            # do not have positive samples in regression loss
            loss_reg = torch.Tensor([0]).sum().to(reg_pred.device)
        else:
            loss_reg = self.reg_loss(sampled_reg_pred, gt_reg)
            loss_reg /= num_pos

        total_mi_zy = spacial_mi + temporal_mi

        losses = {"rpn_cls": loss_cls, "rpn_reg": loss_reg, "mi_zy":total_mi_zy, "loss_spacial":loss_spacial, "loss_temporal":loss_temporal }
        return losses

    @torch.no_grad()
    def prepare_targets(self, anchors, masks, gt_segments, gt_labels):
        # prepare gts: assign the gt_segment to each anchor
        anchors = torch.cat(anchors, dim=0)  # [B,K,2]

        gt_cls_list, gt_reg_list, pos_idxs_list, neg_idxs_list = [], [], [], []
        for i, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
            if len(gt_segment) == 0:  # make a pseudo gt_segment
                gt_segment = torch.tensor([[0, 0]], dtype=torch.float32, device=anchors.device)
                gt_label = torch.zeros(self.num_classes, device=anchors.device).to(torch.int64)
            else:
                gt_label = torch.ones((gt_segment.shape[0], 1), device=anchors.device).to(torch.int64)  # binary label

            # assign GT for valid positions
            _, assigned_gt_idxs, assigned_labels = self.assigner.assign(
                anchors[masks[i]],
                gt_segment,
                gt_label,
            )

            # sample positive and negative anchors
            pos_idxs, neg_idxs = self.sampler.sample(assigned_gt_idxs)
            pos_idxs_list.append(pos_idxs)
            neg_idxs_list.append(neg_idxs)

            # classification target: pos_mask + neg_mask
            gt_cls = assigned_labels[pos_idxs + neg_idxs].squeeze(-1)
            gt_cls_list.append(gt_cls)

            # regression target: pos_mask
            gt_reg = compute_delta(anchors[masks[i]][pos_idxs], gt_segment[assigned_gt_idxs[pos_idxs] - 1])
            gt_reg_list.append(gt_reg)

        gt_cls_concat = torch.cat(gt_cls_list, dim=0)  #  [B*(pos+neg)]
        gt_reg_concat = torch.cat(gt_reg_list, dim=0)  #  [B*(pos),2]
        return gt_cls_concat, gt_reg_concat, pos_idxs_list, neg_idxs_list

    @torch.no_grad()
    def _get_nms_proposal_list(self, anchors, mask_list, cls_pred, reg_pred):
        bs = cls_pred[0].shape[0]
        device = cls_pred[0].device
        pre_nms_topk = 2000
        post_nms_topk = 1000
        nms_thresh = 0.7

        # for each feature map, apply delta_to_pred() and select top-k anchors before nms
        topk_proposals, topk_scores, topk_masks, level_ids = [], [], [], []
        batch_idx = torch.arange(bs, device=device)
        for l, (anchor_i, logits_i, reg_i, mask_i) in enumerate(zip(anchors, cls_pred, reg_pred, mask_list)):
            # 1. get valid anchors
            mask_i = mask_i.unsqueeze(-1).repeat(1, 1, len(self.scales)).flatten(1)  # [bs,T*len(scales)]

            # 2. apply delta_to_pred() to get proposals
            reg_i = reg_i.permute(0, 2, 1).reshape(bs, -1, 2)
            scores_i = logits_i.permute(0, 2, 1).reshape(bs, -1).sigmoid()  # [bs, T*len(scales)]
            proposals_i = delta_to_pred(anchor_i, reg_i)  # [bs, T*len(scales), 2]

            # 3. select top-k anchor for each level and each video
            num_proposals_i = min(proposals_i.shape[1], pre_nms_topk)
            topk_scores_i, topk_idx = scores_i.topk(num_proposals_i, dim=1)
            topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # [bs,topk,2]
            topk_masks_i = mask_i[batch_idx[:, None], topk_idx]  # [bs,topk]

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            topk_masks.append(topk_masks_i)
            level_ids.append(torch.full((num_proposals_i,), l, dtype=torch.int64, device=device))

        # concat all levels together
        topk_proposals = torch.cat(topk_proposals, dim=1)
        topk_scores = torch.cat(topk_scores, dim=1)
        topk_masks = torch.cat(topk_masks, dim=1)
        level_ids = torch.cat(level_ids, dim=0)  # we have recorded the level id

        # NMS on each level, and choose topk results.
        nms_proposals, nms_scores = [], []
        for i in range(bs):
            # select valid proposals
            valid = topk_masks[i]

            # NMS on each feature map
            new_proposals, new_scores, _ = batched_nms(
                topk_proposals[i][valid],
                topk_scores[i][valid],
                level_ids[valid],
                iou_threshold=nms_thresh,
                max_seg_num=post_nms_topk,
                use_soft_nms=False,
                multiclass=True,
            )

            nms_proposals.append(new_proposals.to(device))
            nms_scores.append(new_scores.to(device))
        return nms_proposals, nms_scores

    # @torch.no_grad()
    def _get_proposal_list(self, anchors, mask_list, cls_pred, reg_pred):
        bs = cls_pred[0].shape[0]

        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)  # [B,T*len(scales),2]
        cls_pred = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1)  # [B,T*len(scales)]

        anchors = torch.cat(anchors, dim=0).unsqueeze(0)  # [1,T*len(scales),2]
        proposals = delta_to_pred(anchors, reg_pred.detach())  # [B,K,2]
        masks = torch.cat(mask_list, dim=1).unsqueeze(-1).repeat(1, 1, len(self.scales))  # [B,T,len(scales)]

        new_proposals, new_scores = [], []
        for proposal, logits, mask in zip(proposals, cls_pred, masks):
            new_proposals.append(proposal[mask.view(-1)])
            new_scores.append(logits[mask.view(-1)].detach().sigmoid())
        return new_proposals, new_scores


class CLUBMulti(nn.Module):
    def __init__(self, z_channels, temporal_dim=2, hidden_channels=128, num_classes=200):
        super().__init__()
        # Define mu_net and logvar_net for each group

        # 为每个尺度定义独立的条件网络
        # 尺度1
        self.spacial_net_1 = CLUBForCategorical(z_channels[0], num_classes, hidden_size=num_classes // 2)
        self.temporal_net_1 = CLUB_ZY(z_channels[0], temporal_dim, hidden_size=z_channels[0] // 2)

        self.spacial_net_2 = CLUBForCategorical(z_channels[1], num_classes, hidden_size=num_classes // 2)
        self.temporal_net_2 = CLUB_ZY(z_channels[1], temporal_dim, hidden_size=z_channels[1] // 2)

        self.spacial_net_3 = CLUBForCategorical(z_channels[2], num_classes, hidden_size=num_classes // 2)
        self.temporal_net_3 = CLUB_ZY(z_channels[2], temporal_dim, hidden_size=z_channels[2] // 2)

        self.spacial_net_4 = CLUBForCategorical(z_channels[3], num_classes, hidden_size=num_classes // 2)
        self.temporal_net_4 = CLUB_ZY(z_channels[3], temporal_dim, hidden_size=z_channels[3] // 2)

        self.spacial_net_5 = CLUBForCategorical(z_channels[4], num_classes, hidden_size=num_classes // 2)
        self.temporal_net_5 = CLUB_ZY(z_channels[4], temporal_dim, hidden_size=z_channels[4] // 2)

    def forward_once(self, x, z, gt_segments=None, gt_labels=None, spacial_net=None, temporal_net=None, input_net=None):
        mi_spacial, loss_spacial = spacial_net(z, gt_labels)
        mi_temporal, loss_temporal = temporal_net(z, gt_segments)
        mi_input, loss_input = input_net(x, z)
        mi_estimate = mi_spacial + mi_temporal - 0.25 * mi_input
        loss = loss_spacial + loss_temporal + loss_input
        return mi_estimate, loss

    def forward(self, x, z, gt_segments=None, gt_labels=None):
        # Perform forward_once for each pair of mu_net and logvar_net
        upperbound_1, loss_1 = self.forward_once(x, z[0], gt_segments, gt_labels, self.spacial_net_1,
                                                 self.temporal_net_1)
        # print(upperbound_1, loss_1)
        upperbound_2, loss_2 = self.forward_once(x, z[1], self.mu_net_2, self.logvar_net_2)
        # print(upperbound_2, loss_2)
        upperbound_3, loss_3 = self.forward_once(x, z[2], self.mu_net_3, self.logvar_net_3)
        # print(upperbound_3, loss_3)
        upperbound_4, loss_4 = self.forward_once(x, z[3], self.mu_net_4, self.logvar_net_4)
        # print(upperbound_4, loss_4)
        upperbound_5, loss_5 = self.forward_once(x, z[4], self.mu_net_5, self.logvar_net_5)

        # Compute the average upperbound and loss
        avg_upperbound = (0.1 * upperbound_1 + 0.2 * upperbound_2 + 0.3 * upperbound_3 + 0.4 * upperbound_4 + 0.5 * upperbound_5) / 1.5
        avg_loss = (0.1 * loss_1 + 0.2 * loss_2 + 0.3 * loss_3 + 0.4 * loss_4 + 0.5 * loss_5) / 1.5

        return avg_upperbound, avg_loss


class CLUBForCategorical(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels
        '''
        super().__init__()
        self.label_dim = label_dim
        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_dim)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_dim)
            )

    def forward(self, inputs, labels):
        logits = self.variational_net(inputs.transpose(1, 2))  # [sample_size, label_num]
        print(logits.shape)  ## [batch_size, lenght, max_class]

        # 计算交叉熵时应用mask
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        ).reshape(sample_size, sample_size) * mask_extend.squeeze(1)

        mi_estimate = positive - negative
        loss = self.learning_loss(inputs, labels)
        return mi_estimate, loss

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)

    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)


class CLUB_ZY(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, z_dim, y_dim, hidden_size):
        super(CLUB_ZY, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(z_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(z_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, z):
        noise = torch.randn_like(z).cuda()
        z_samples = z + noise
        # x_samples = torch.mean(x_samples.mean(dim=1),dim=1)
        mu = self.p_mu(z_samples)
        logvar = self.p_logvar(z_samples)
        return mu, logvar

    def forward(self, z_samples, y_samples):
        mu, logvar = self.get_mu_logvar(z_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()
        mi_estimate = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        loss = self.learning_loss(z_samples, y_samples)

        return mi_estimate, loss

    def loglikeli(self, z_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(z_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, z_samples, y_samples):
        return - self.loglikeli(z_samples, y_samples)
