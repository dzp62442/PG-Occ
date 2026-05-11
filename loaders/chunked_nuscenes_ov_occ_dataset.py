import copy
import io
import json
import math
import os
import random
import time
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmdet.datasets import DATASETS
from mmcv.runner import get_dist_info
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import IterableDataset, get_worker_info

from .ego_pose_dataset import EgoPoseDataset
from .old_metrics import main_miou
from .ray_metrics import main_rayiou, main_raypq
from configs.pgocc import occ_class_names as occ3d_class_names


PGOCC_CAM_TYPES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def torch_load_bytes(data):
    buffer = io.BytesIO(data)
    try:
        return torch.load(buffer, map_location='cpu', weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location='cpu')


@DATASETS.register_module()
class NuSceneOVOccChunk(IterableDataset):
    """PG-Occ nuScenes OV occupancy dataset backed by GaussTR chunks."""

    is_chunk_iterable = True

    def __init__(self,
                 chunk_root='data/gausstr_chunks',
                 split='train',
                 profile='featup_metric3d_sam2',
                 ann_file=None,
                 data_root='data/nuscenes/',
                 occ_gt_root='data/nuscenes/gts',
                 pipeline=None,
                 metric=None,
                 modality=None,
                 return_intrinsic=True,
                 render_conf=None,
                 chunk_shuffle=True,
                 sample_shuffle=True,
                 seed=2026,
                 pad_train_chunks=True,
                 skip_padding=True,
                 load_to_memory=False,
                 debug=False,
                 test_mode=False,
                 **kwargs):
        super().__init__()
        self.chunk_root = Path(chunk_root)
        self.split = split
        self.profile = profile
        self.profile_root = self.chunk_root / split / profile
        self.ann_file = ann_file
        self.data_root = data_root
        self.occ_gt_root = occ_gt_root
        self.metric = metric or ['miou']
        self.modality = modality or dict(use_camera=True)
        self.return_intrinsic = return_intrinsic
        self.render_conf = render_conf or {}
        self.chunk_shuffle = bool(chunk_shuffle)
        self.sample_shuffle = bool(sample_shuffle)
        self.seed = int(seed)
        self.pad_train_chunks = bool(pad_train_chunks)
        self.skip_padding = bool(skip_padding)
        self.load_to_memory = bool(load_to_memory)
        self.debug = bool(debug)
        self.test_mode = bool(test_mode)
        self._epoch = 0
        self._iter_epoch = 0

        self.pipeline = self._build_pipeline(pipeline or [])
        self.manifest = self._load_json(self.profile_root / 'chunk_manifest.json')
        self.index = self._load_json(self.profile_root / 'index.json')
        self.profile_meta = self._load_json(self.profile_root / 'profile.json')
        self.samples_per_chunk = int(self.manifest.get('samples_per_chunk', 1))
        self.chunks = [
            dict(chunk_id=chunk_id, **entry)
            for chunk_id, entry in sorted(self.manifest['chunks'].items())
        ]
        self.index_samples = self._index_samples()
        self.num_valid_samples = len(self.index_samples)
        self.expected_source_indices = [
            int(item['sample_idx']) for item in self.index_samples
        ]
        self.expected_chunk_sample_idx = [
            str(item['sample_idx']) for item in self.index_samples
        ]
        self._selected_offsets_by_chunk = self._group_offsets_by_chunk(
            self.index_samples)
        self.data_infos = self._load_data_infos(ann_file)
        self.eval_infos = self._build_eval_infos()
        self.expected_tokens = [info['token'] for info in self.eval_infos]
        self.flag = np.zeros(self.num_valid_samples, dtype=np.uint8)

    @staticmethod
    def _build_pipeline(pipeline):
        from mmdet3d.datasets.pipelines import Compose
        return Compose(pipeline)

    @staticmethod
    def _load_json(path):
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def _load_data_infos(self, ann_file):
        if ann_file is None:
            return []
        ann = mmcv.load(ann_file)
        return ann['infos'] if isinstance(ann, dict) and 'infos' in ann else ann

    def _index_samples(self):
        samples = self.index.get('samples')
        if samples:
            return sorted(samples, key=lambda item: int(item['global_offset']))
        by_sample_idx = self.index.get('by_sample_idx', {})
        items = []
        for sample_idx, entry in by_sample_idx.items():
            items.append(dict(entry, sample_idx=str(sample_idx)))
        return sorted(items, key=lambda item: int(item['global_offset']))

    @staticmethod
    def _group_offsets_by_chunk(index_samples):
        grouped = {}
        for item in index_samples:
            grouped.setdefault(str(item['chunk_id']), set()).add(int(item['offset']))
        return grouped

    def _build_eval_infos(self):
        if not self.data_infos:
            return []
        infos = []
        for source_index in self.expected_source_indices:
            if source_index >= len(self.data_infos):
                raise IndexError(
                    f'Chunk source_index {source_index} exceeds ann_file '
                    f'length {len(self.data_infos)}.')
            infos.append(self.data_infos[source_index])
        return infos

    def __len__(self):
        rank, world_size = get_dist_info()
        if self.split == 'train':
            if self.pad_train_chunks and self.chunks:
                return math.ceil(len(self.chunks) / world_size) * self.samples_per_chunk
            return len(self._partition_chunks(self._ordered_chunks(self._epoch)))
        return self.num_valid_samples

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def _current_epoch(self):
        epoch = self._epoch + self._iter_epoch
        self._iter_epoch += 1
        return epoch

    def _ordered_chunks(self, epoch):
        chunks = list(self.chunks)
        if self.split == 'train' and self.chunk_shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(chunks)
        if self.split == 'train' and self.pad_train_chunks and chunks:
            _, world_size = get_dist_info()
            total = math.ceil(len(chunks) / world_size) * world_size
            for i in range(total - len(chunks)):
                chunks.append(chunks[i % len(chunks)])
        return chunks

    @staticmethod
    def _partition_list(items):
        rank, world_size = get_dist_info()
        items = items[rank::world_size]
        worker = get_worker_info()
        if worker is not None:
            items = items[worker.id::worker.num_workers]
        return items

    def _partition_chunks(self, chunks):
        return self._partition_list(chunks)

    def _eval_chunk_offsets(self):
        rank, world_size = get_dist_info()
        worker = get_worker_info()
        selected = [
            item for item in self.index_samples
            if int(item['global_offset']) % world_size == rank
        ]
        if worker is not None:
            selected = selected[worker.id::worker.num_workers]
        grouped = {}
        for item in selected:
            grouped.setdefault(str(item['chunk_id']), set()).add(int(item['offset']))
        chunks = [
            self._chunk_by_id()[chunk_id] for chunk_id in sorted(
                grouped, key=lambda cid: int(cid))
        ]
        return [(chunk, grouped[str(chunk['chunk_id'])]) for chunk in chunks]

    def _chunk_by_id(self):
        return {str(chunk['chunk_id']): chunk for chunk in self.chunks}

    def _chunk_path(self, entry):
        return self.profile_root / entry['path']

    def _load_chunk(self, entry):
        path = self._chunk_path(entry)
        start = time.monotonic()
        payload = torch_load(path)
        if self.debug:
            print(
                f'[NuSceneOVOccChunk] loaded chunk={entry["chunk_id"]} '
                f'elapsed={time.monotonic() - start:.2f}s path={path}',
                flush=True)
        return payload

    def _iter_samples(self, chunk, payload, epoch, valid_offsets=None):
        indexed = list(enumerate(payload['samples']))
        if self.split == 'train' and self.sample_shuffle:
            rng = random.Random(self.seed + epoch * 1000003 + int(chunk['chunk_id']))
            rng.shuffle(indexed)
        for offset, sample in indexed:
            if valid_offsets is not None and offset not in valid_offsets:
                continue
            if sample.get('is_padding', False) and self.skip_padding:
                continue
            results = self._sample_to_results(sample)
            data = self.pipeline(results)
            if data is not None:
                yield data

    def __iter__(self):
        epoch = self._current_epoch()
        if self.split == 'train':
            chunks = self._partition_chunks(self._ordered_chunks(epoch))
            for chunk in chunks:
                yield from self._iter_samples(chunk, self._load_chunk(chunk), epoch)
            return

        for chunk, offsets in self._eval_chunk_offsets():
            yield from self._iter_samples(
                chunk, self._load_chunk(chunk), epoch, valid_offsets=offsets)

    def _ann_info_for_sample(self, sample):
        source_index = int(sample.get('source_index', sample.get('sample_idx')))
        if self.data_infos:
            return self.data_infos[source_index]
        return None

    def _sample_to_results(self, sample):
        info = self._ann_info_for_sample(sample)
        if info is None:
            raise RuntimeError('NuSceneOVOccChunk requires ann_file for PG-Occ metadata.')

        ego2global_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        lidar2ego_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        ego2lidar = transform_matrix(
            info['lidar2ego_translation'],
            Quaternion(info['lidar2ego_rotation']),
            inverse=True)

        input_dict = dict(
            sample_idx=info['token'],
            chunk_sample_idx=str(sample.get('source_sample_idx', sample.get('sample_idx'))),
            source_index=int(sample.get('source_index', sample.get('sample_idx'))),
            token=info['token'],
            scene_name=info['scene_name'],
            sweeps={'prev': [], 'next': []},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=lidar2ego_rotation,
            ego2lidar=[ego2lidar for _ in PGOCC_CAM_TYPES],
            occ_path=os.path.join(
                self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz'),
            _chunk_sample=sample,
            _chunk_camera_order=list(PGOCC_CAM_TYPES),
        )

        if self.modality.get('use_camera', True):
            img_paths, img_timestamps = [], []
            lidar2img_rts, ego2img_rts, cam2ego_rts = [], [], []
            feature_names, ori_ks, img_auxi_paths, cam2global_rts = [], [], [], []

            for cam in PGOCC_CAM_TYPES:
                if cam not in info['cams']:
                    raise KeyError(f'ann_file sample {info["token"]} missing camera {cam}.')
                if cam not in sample['image_bytes']:
                    raise KeyError(f'chunk sample {sample.get("sample_idx")} missing camera {cam}.')
                cam_info = info['cams'][cam]
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_auxi_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                cam2ego = np.eye(4)
                cam2ego[:3, :3] = cam_info['sensor2ego_rotation']
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2ego_rts.append(cam2ego)
                cam2global_rts.append(cam_info['cam2global'])

                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rts.append(viewpad @ lidar2cam_rt.T)
                ego2img_rts.append(viewpad @ np.linalg.inv(cam2ego))
                ori_ks.append(intrinsic)
                feature_names.append(os.path.splitext(os.path.basename(
                    cam_info['data_path']))[0] + '.npy')

            input_dict.update(dict(
                img_filename=img_paths,
                filename=img_paths,
                feature_names=feature_names,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                ego2img=ego2img_rts,
                cam2ego=cam2ego_rts,
                cam2global=cam2global_rts,
                img_auxi_paths=img_auxi_paths,
                ori_k=ori_ks,
            ))

        return input_dict

    @staticmethod
    def _prediction_to_numpy(prediction, expected_shape, token):
        if isinstance(prediction, torch.Tensor):
            pred = prediction.detach().cpu().numpy()
        else:
            pred = np.asarray(prediction)

        while pred.ndim > len(expected_shape) and pred.shape[0] == 1:
            pred = pred[0]

        if pred.shape != expected_shape:
            raise ValueError(
                f'Chunk prediction shape mismatch for token {token}: '
                f'pred_shape={pred.shape}, expected_shape={expected_shape}')
        return pred

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        print('\nStarting Chunk Evaluation...')
        results = {}
        pred_by_token = {}
        duplicate_tokens = []

        for result in occ_results:
            token = result.get('sample_idx') or result.get('token')
            if token is None:
                raise KeyError('Chunk evaluation result missing sample_idx/token.')
            token = str(token)
            if token in pred_by_token:
                duplicate_tokens.append(token)
            pred_by_token[token] = result

        expected = list(self.expected_tokens)
        missing = [token for token in expected if token not in pred_by_token]
        unexpected = sorted(set(pred_by_token) - set(expected))
        if duplicate_tokens or missing or unexpected:
            raise RuntimeError(
                'Chunk evaluation coverage failed: '
                f'duplicates={duplicate_tokens[:10]}, '
                f'missing={missing[:10]}, unexpected={unexpected[:10]}')

        occ_gts, occ_preds, inst_gts, inst_preds = [], [], [], []
        lidar_origins, cam_masks = [], []

        if 'depth' in self.metric:
            error = {cam_type: [] for cam_type in PGOCC_CAM_TYPES}
            for token in expected:
                occ_result = pred_by_token[token]
                for cam_idx, error_value in enumerate(occ_result['depth_error']):
                    error[PGOCC_CAM_TYPES[cam_idx]].append(error_value)
            mean_errors = []
            for cam in PGOCC_CAM_TYPES:
                mean_error = np.nanmean(np.array(error[cam]), axis=0)
                mean_errors.append(mean_error)
                results[f'{cam}'] = mean_error
            results['average'] = np.nanmean(np.stack(mean_errors), axis=0)

        token_to_info = {info['token']: info for info in self.eval_infos}
        for batch in torch.utils.data.DataLoader(
                EgoPoseDataset(self.eval_infos), num_workers=0):
            token = batch[0][0]
            output_origin = batch[1]
            info = token_to_info[token]
            occ_path = os.path.join(
                self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            gt_mask_camera = occ_gt['mask_camera'].astype(bool)
            occ_pred = pred_by_token[token]
            sem_pred = self._prediction_to_numpy(
                occ_pred['occ_preds'], gt_semantics.shape, token)

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
            cam_masks.append(gt_mask_camera)

        data_type = self.occ_gt_root.split('/')[-1]
        if data_type not in ['gts', 'occ3d_panoptic']:
            raise ValueError(f'Unsupported dataset type: {data_type}')
        occ_class_names = occ3d_class_names

        if len(inst_preds) > 0:
            results.update(main_raypq(
                occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins,
                occ_class_names=occ_class_names))
            results.update(main_rayiou(
                occ_preds, occ_gts, lidar_origins,
                occ_class_names=occ_class_names))
        else:
            if 'rayiou' in self.metric:
                results.update(main_rayiou(
                    occ_preds, occ_gts, lidar_origins,
                    occ_class_names=occ_class_names))
            if 'miou' in self.metric:
                results.update(main_miou(
                    occ_preds, occ_gts, cam_masks,
                    occ_class_names=occ_class_names))
        return results
