_base_ = './pgocc.py'

dataset_type = 'NuSceneOVOccChunk'
dataset_root = 'data/nuscenes/'
occ_gt_root = 'data/nuscenes/gts'
chunk_root = 'data/gausstr_chunks'
chunk_profile = 'featup_metric3d_sam2'

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

render_conf = dict(
    use_ov=True,
    render_h=180,
    render_w=320,
    ov_auxi_past_frame_num=0,
    ov_auxi_future_frame_num=0,
)

_metric_ = ['miou']
_return_intrinsic_ = True

loss_weights = dict(
    depth_warping=0.0,
    ov_mse=10.0,
    ov_cos=1.0,
    depth_foundation=0.5,
    depth_gt=0.5,
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': False,
    'use_actual_shape': True,
}

model = dict(
    pts_bbox_head=dict(
        loss_weights=loss_weights,
        transformer=dict(num_frames=1)))

train_pipeline = [
    dict(type='LoadMultiViewImageFromChunks', to_float32=False, color_type='color'),
    dict(type='LoadChunkFeature', key='depth', out_key='depth'),
    dict(type='LoadChunkFeature', key='feats', out_key='text_vision'),
    dict(type='GenerateCurrentRenderImage', render_conf=render_conf),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(
        type='Collect3D',
        keys=['img', 'text_vision', 'depth', 'render_gt'],
        meta_keys=(
            'sample_idx', 'token', 'chunk_sample_idx', 'source_index',
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2img',
            'img_timestamp', 'cam2ego', 'ego2lidar', 'lidar2img',
            'render_k', 'ori_k', 'scene_name'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromChunks', to_float32=False, color_type='color'),
    dict(type='LoadChunkFeature', key='depth', out_key='depth'),
    dict(type='LoadChunkFeature', key='feats', out_key='text_vision'),
    dict(type='LoadChunkOccGT', num_classes=len(occ_class_names)),
    dict(type='GenerateCurrentRenderImage', render_conf=render_conf),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(
        type='Collect3D',
        keys=['mask_camera', 'img', 'voxel_semantics', 'text_vision', 'depth', 'render_gt'],
        meta_keys=(
            'sample_idx', 'token', 'chunk_sample_idx', 'source_index',
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 'ego2img',
            'img_timestamp', 'cam2ego', 'ego2lidar', 'lidar2img',
            'render_k', 'ori_k', 'scene_name'))
]

data = dict(
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type=dataset_type,
        chunk_root=chunk_root,
        split='train',
        profile=chunk_profile,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        metric=_metric_,
        pipeline=train_pipeline,
        modality=input_modality,
        return_intrinsic=_return_intrinsic_,
        render_conf=render_conf,
        chunk_shuffle=True,
        sample_shuffle=True,
        pad_train_chunks=True,
        skip_padding=False,
        test_mode=False),
    val=dict(
        _delete_=True,
        type=dataset_type,
        chunk_root=chunk_root,
        split='val',
        profile=chunk_profile,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        metric=_metric_,
        pipeline=test_pipeline,
        modality=input_modality,
        return_intrinsic=_return_intrinsic_,
        render_conf=render_conf,
        chunk_shuffle=False,
        sample_shuffle=False,
        pad_train_chunks=False,
        skip_padding=True,
        test_mode=True),
    test=dict(
        _delete_=True,
        type=dataset_type,
        chunk_root=chunk_root,
        split='val',
        profile=chunk_profile,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        metric=_metric_,
        pipeline=test_pipeline,
        modality=input_modality,
        return_intrinsic=_return_intrinsic_,
        render_conf=render_conf,
        chunk_shuffle=False,
        sample_shuffle=False,
        pad_train_chunks=False,
        skip_padding=True,
        test_mode=True),
)

eval_config = dict(interval=8)
debug = True
