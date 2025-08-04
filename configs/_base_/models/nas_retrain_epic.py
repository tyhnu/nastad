model = dict(
    type="NASRetrain",
    projection=dict(
        type="NASProjRetrain",
        in_channels=2048,
        out_channels=512,
        super_arch=(2, 15, 5),  
        choice=[0, 1, 0, 2, 1, 1, 2, 2, 1, 0, 0, 0, 2, 1, 1, 1, 1, 2, 0, 2],  ## searched_network_arch
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        use_abs_pe=True,
        max_seq_len=2304,
        mamba_kernel_size=4,
        channel_expand=2,
        num_head=4,
        drop_path_rate=0.3,
        input_pdrop=0.2,

    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=293,  # total 300, but 7 classes are empty
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=250,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.1,
        loss_weight=0.5,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)
