"""Small eight-rank NCCL initialization/all-reduce check without model loading."""
import os
import torch
import torch.multiprocessing as mp


def worker(rank):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' if rank < 4 else '4,5,6,7'
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    local = rank % 4
    torch.cuda.set_device(local)
    role_group = StatelessProcessGroup.create('127.0.0.1', 29880 + rank // 4, local, 4)
    role_communicator = PyNcclCommunicator(role_group, device=local)
    group = StatelessProcessGroup.create('127.0.0.1', 29890 + local, rank // 4, 2)
    communicator = PyNcclCommunicator(group, device=local)
    communicator.disabled = False
    tensor = torch.tensor([float(rank)], device=local)
    output = communicator.all_reduce(tensor)
    torch.cuda.synchronize()
    assert output.item() == local * 2 + 4, (rank, output.item())
    print(f'PASS rank={rank} sum={output.item()}', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    mp.spawn(worker, nprocs=8, join=True)
