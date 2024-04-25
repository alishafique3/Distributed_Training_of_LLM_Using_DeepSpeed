# For all details
# https://pytorch.org/tutorials/intermediate/dist_tuto.html#distributed-training
# https://pytorch.org/docs/stable/distributed.html

"""run.py:"""
#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# def run(rank, size):
#     """ Distributed function to be implemented later. """
#     pass

# """Blocking point-to-point communication."""
# def run(rank, size):
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])

# """Non-blocking point-to-point communication."""
# def run(rank, size):
#     tensor = torch.zeros(1)
#     req = None
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         req = dist.isend(tensor=tensor, dst=1)
#         print('Rank 0 started sending')
#     else:
#         # Receive tensor from process 0
#         req = dist.irecv(tensor=tensor, src=0)
#         print('Rank 1 started receiving')
#     req.wait()
#     print('Rank ', rank, ' has data ', tensor[0])
    
""" All-Reduce example."""
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Rank ', rank, ' has data ', tensor[0], 'on device', device)
    

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)



if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()