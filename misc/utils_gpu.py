import torch 
from time import perf_counter 

def get_device(gpu_idx=0):
    device = "cpu"
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        #print("%i available GPUs" % available_gpus)
        while gpu_idx > available_gpus-1:
            gpu_idx = gpu_idx - 1
        device = torch.device("cuda:%i"%gpu_idx)
    #print("Device: ", device)    
    return device

class MemoryLogger:

    def __init__(self, printing=True):
        if not torch.cuda.is_available():
            print("CUDA not available!")
            return None
        self.mem_ids = {}
        self.printing=printing

    def start_id(self, id, gpu_idx=0):
        if not torch.cuda.is_available():
            return None
            
        a = torch.cuda.memory_allocated(gpu_idx)
        t = perf_counter()
        id = id + ":" + str(gpu_idx)
        self.mem_ids[id] = [a, t]

    def end_id(self, id, gpu_idx=0):
        if not torch.cuda.is_available():
            return None
            
        id = id + ":" + str(gpu_idx)
        if id not in self.mem_ids.keys():
            print("No key with ID %s found!" % id)
            return None
        total_allocated = torch.cuda.memory_allocated(gpu_idx) / 1000 / 1000
        new_allocated = (torch.cuda.memory_allocated(gpu_idx) - self.mem_ids[id][0]) / 1000 / 1000
        time = perf_counter() - self.mem_ids[id][1]
        # log 
        if self.printing:
            print("[GPU:%i] %s: %.2fMB new memory allocated (Total: %.2f), in %.2f seconds" % (gpu_idx, id, new_allocated, total_allocated, time))
        return time
        


if __name__ == '__main__':
    print("Available GPUs: %i" % torch.cuda.device_count())
    get_device(gpu_idx=1)
