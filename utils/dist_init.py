import os
import torch

class Utils:

    world_size = torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])

    @staticmethod
    def initialize_distributed():
        print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
        torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(backend='nccl', world_size=Utils.world_size, rank=Utils.rank, init_method=init_method)
        