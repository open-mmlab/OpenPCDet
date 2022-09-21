import torch
import slice_and_batch_cuda
import time

channels=3
inp = torch.rand((channels,5,5)).cuda()
print('Input', inp.stride(), ':\n', inp)
slice_indices=torch.Tensor([1,2]).type(torch.long).cuda()
print(slice_indices.type())
slice_size=3
outp = torch.zeros((slice_indices.size()[0], channels, slice_size, slice_size)).cuda()
slice_and_batch_cuda.slice_and_batch_cuda(inp, slice_indices, slice_size, outp)
print('Output:\n', outp)

#times_ms=[]
#for i in range(101):
#    inp = torch.rand((1,64,400,352)).cuda()
#    hm = torch.rand((1,3,400,352)).cuda() / 9
#    torch.cuda.synchronize()
#    t1 = time.time()
#    slices = slice_and_batch.slice_and_batch(inp, hm, 500, 5, 0.1)
#    torch.cuda.synchronize()
#    t2 = time.time()
#    times_ms.append((t2-t1)*1000)
##    print('Num slices:', slices[1])
#times_ms = times_ms[1:]
##print(times_ms)
#print(sum(times_ms)/len(times_ms))
