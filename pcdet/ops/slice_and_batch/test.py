import torch
import slice_and_batch
import time

times_ms=[]
for i in range(101):
    inp = torch.rand((1,64,400,352)).cuda()
    hm = torch.rand((1,3,400,352)).cuda() / 9
    torch.cuda.synchronize()
    t1 = time.time()
    slices = slice_and_batch.slice_and_batch(inp, hm, 500, 5, 0.1)
    torch.cuda.synchronize()
    t2 = time.time()
    times_ms.append((t2-t1)*1000)
#    print('Num slices:', slices[1])
times_ms = times_ms[1:]
#print(times_ms)
print(sum(times_ms)/len(times_ms))
