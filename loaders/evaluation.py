import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.core import DistEvalHook
from mmdet.core.mask import encode_mask_results
from torch.nn.modules.batchnorm import _BatchNorm


def is_chunk_dataset(dataset):
    return getattr(dataset, 'is_chunk_iterable', False)


def _collect_results_gpu_concat(result_part):
    """Collect uneven rank-local result lists without zip truncation."""
    rank, world_size = get_dist_info()

    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    dist.all_gather(part_recv_list, part_send)

    if rank != 0:
        return None

    results = []
    for recv, shape in zip(part_recv_list, shape_list):
        part = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
        results.extend(part)
    return results


def _collect_results_cpu_concat(result_part, tmpdir=None):
    """CPU variant of uneven result collection for chunk iterable datasets."""
    rank, world_size = get_dist_info()
    if tmpdir is None:
        max_len = 512
        dir_tensor = torch.full(
            (max_len,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir_tensor = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir_tensor)] = tmpdir_tensor
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank != 0:
        return None

    results = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        results.extend(mmcv.load(part_file))
    shutil.rmtree(tmpdir)
    return results


def chunk_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Multi-GPU test for internally sharded chunk IterableDatasets.

    MMDet's default result collection interleaves ranks with ``zip(*parts)``
    and truncates to ``len(dataset)``. That assumes sampler-style padding and
    equal rank-local result counts. Chunk datasets shard internally and can
    produce uneven rank counts, so collection must concatenate all rank-local
    results and let dataset.evaluate validate sample coverage by token.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        progress_count = 0
    time.sleep(2)

    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            update_count = min(len(dataset) - progress_count,
                               len(result) * world_size)
            progress_count += update_count
            for _ in range(update_count):
                prog_bar.update()

    if gpu_collect:
        return _collect_results_gpu_concat(results)
    return _collect_results_cpu_concat(results, tmpdir)


class ChunkDistEvalHook(DistEvalHook):
    """Dist eval hook that uses chunks-aware uneven result collection."""

    def _do_evaluate(self, runner):
        if self.broadcast_bn_buffer:
            model = runner.model
            for _, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = chunk_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
