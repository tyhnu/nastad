import torch.nn as nn
import torch
import torch.nn.functional as F
from .single_stage import SingleStageDetector
from ..builder import DETECTORS
from ..bricks import Scale, AffineDropPath
from ..utils.post_processing import load_predictions, save_predictions
import numpy as np

@DETECTORS.register_module()
class VideoMambaSuite(SingleStageDetector):
    def __init__(self, projection, rpn_head, neck=None, backbone=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)
                elif (
                    pn.endswith("A_log")
                    or pn.endswith("D_b")
                    or pn.endswith("D")
                    or pn.endswith("A_b_log")
                    or pn.endswith("forward_embed")
                    or pn.endswith("backward_embed")
                ):
                    # corner case for mamba
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups


@DETECTORS.register_module()
class VideoMambaSuiteNAS(SingleStageDetector):
    def __init__(self, projection, rpn_head, neck=None, backbone=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)
                elif (
                    pn.endswith("A_log")
                    or pn.endswith("D_b")
                    or pn.endswith("D")
                    or pn.endswith("A_b_log")
                    # or pn.endswith("lepe")
                    or pn.endswith("forward_embed")
                    or pn.endswith("backward_embed")
                ):
                    # corner case for mamba
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks, choice = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_losses = self.rpn_head.forward_train(
                x,
                masks,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
            losses.update(rpn_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses, choice

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, input_choice=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs
        # print(input_choice)
        if self.with_projection:
            x, masks,choice = self.projection(x, masks,input_choice)
        # assert np.array_equal(input_choice, choice)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks)
        else:
            rpn_proposals = rpn_scores = None

        predictions = rpn_proposals, rpn_scores
        return predictions

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, input_choice=None, **kwargs):
        # step1: inference the model
        # print(inputs)
        # print(input_choice)
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg, input_choice=input_choice)

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results

@DETECTORS.register_module()
class VideoMambaSuiteNASIB(SingleStageDetector):
    def __init__(self, projection, rpn_head, neck=None, backbone=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )
    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # 参数分组逻辑：决定哪些参数应用 weight decay，哪些不应用
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # 获取完整参数名称

                # 排除 backbone 的参数
                if fpn.startswith("backbone"):
                    continue

                # 不进行 weight decay 的情况：
                if pn.endswith("bias"):
                    # 所有 bias 不进行 weight decay
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # 自定义的 Scale 和 AffineDropPath 层的 scale 参数
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # 相对位置编码参数不进行 weight decay
                    no_decay.add(fpn)

                # 进行 weight decay 的情况（白名单模块）：
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # 白名单模块（如 Linear、Conv1d、Conv2d）的 weight 参数
                    decay.add(fpn)

                # 特殊情况处理（Mamba 模型相关参数）：
                elif (
                    pn.endswith("A_log")
                    or pn.endswith("D_b")
                    or pn.endswith("D")
                    or pn.endswith("A_b_log")
                    or pn.endswith("forward_embed")
                    or pn.endswith("backward_embed")
                ):
                    # Mamba 模型中的特殊参数，需要进行 weight decay
                    decay.add(fpn)

                # 黑名单模块的 weight 参数不进行 weight decay
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        total_mi = dict()
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks, choice = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)  # z_list: 多尺度特征列表 [z0, z1, z2]
        else:
            x = [x]  # 单尺度特征

        # 保留原始检测任务逻辑
        if self.with_rpn_head:
            rpn_losses,mi = self.rpn_head.forward_train(inputs,
                x, masks,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs
            )
            losses.update(rpn_losses)
            total_mi.update(mi)


        losses["cost"] = sum(_value for _key, _value in losses.items())
        total_mi["mi"] = sum(_value for _key, _value in total_mi.items())
        return losses, choice, total_mi

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, gt_segments=None, gt_labels=None, input_choice=None, **kwargs):
        # gt_segments and gt_labels are only used for MI evaluation, not for detection
        total_mi = dict()
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs
        # print(input_choice)
        if self.with_projection:
            x, masks,choice = self.projection(x, masks,input_choice)
        # assert np.array_equal(input_choice, choice)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        if self.with_rpn_head:
            rpn_proposals, rpn_scores, mi = self.rpn_head.forward_test(inputs, x, masks, gt_segments, gt_labels)
            total_mi.update(mi)
        else:
            rpn_proposals = rpn_scores = mi = None

        predictions = rpn_proposals, rpn_scores
        total_mi["mi"] = sum(_value for _key, _value in total_mi.items())

        return predictions, total_mi

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, input_choice=None, gt_segments=None, gt_labels=None, **kwargs):
        # step1: inference the model
        # gt_segments and gt_labels are only used for MI evaluation, not for detection
        # print(inputs)
        # print(input_choice)
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions, mi = self.forward_test(inputs, masks, metas, infer_cfg, gt_segments, gt_labels, input_choice=input_choice )

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            results = self.post_processing(predictions, metas, post_cfg, **kwargs)
            return results
        else:
            results = self.post_processing(predictions, metas, post_cfg, **kwargs)
            return results, mi


    def forward(
        self,
        inputs,
        masks,
        metas,
        gt_segments=None,
        gt_labels=None,
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
        else:
            return self.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs) # gt_segments and gt_labels are only used for MI evaluation, not for detection
