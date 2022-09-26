import tensorrt as trt
import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as cuda
import nvtx

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    INPUT_NAME = "inp_tensor"
    OUTPUT_NAME = "outp_tensor"
    DTYPE = trt.float32
    #DTYPE = trt.float16
    W = 128 # 352
    H = 128 # 400

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

@nvtx.annotate("do_inference_v2", color="purple")
def do_inference_v2(context, bindings, inputs, outputs, stream, time_arr):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    stream.synchronize()
    # Run inference.
    t1 = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()
    t2 = time.time()
    time_arr.append((t2-t1)*1000)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# DEFINE NETWORK
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
float_type='float32'
#float_type='float16'

# first part of the centerhead

# 7.125 ms on Jetson AGX Orin
def def_conv_network(network):
    input_tensor = network.add_input(name=ModelData.INPUT_NAME,
            dtype=ModelData.DTYPE, shape=(1,64,ModelData.H,ModelData.W))

    for i in range(5):
        random_weights=np.random.rand(64, 64, 3, 3).astype(float_type)-0.5
        conv = network.add_convolution_nd(input=input_tensor, num_output_maps=64,
                kernel_shape=(3, 3), kernel=random_weights)
        conv.stride_nd=(1,1)
        conv.padding_nd=(1,1)
        relu = network.add_activation(input=conv.get_output(0),
                type=trt.ActivationType.RELU)
        relu.get_output(0).name = ModelData.OUTPUT_NAME + str(i)
        network.mark_output(tensor=relu.get_output(0))

# 6.595 ms on Jetson AGX Orin
def def_merged_conv_network(network):
    input_tensor = network.add_input(name=ModelData.INPUT_NAME,
            dtype=ModelData.DTYPE, shape=(1,64,ModelData.H,ModelData.W))

    random_weights=np.random.rand(320, 64, 3, 3).astype(float_type)-0.5
    conv = network.add_convolution_nd(input=input_tensor, num_output_maps=320,
            kernel_shape=(3, 3), kernel=random_weights)
    conv.stride_nd=(1,1)
    conv.padding_nd=(1,1)
    relu = network.add_activation(input=conv.get_output(0),
            type=trt.ActivationType.RELU)

    relu.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=relu.get_output(0))


# second part of the centerpoint

def def_multi_conv_network(network):
    out_maps = [3,2,1,3,2]
    for i in range(5):
        input_tensor = network.add_input(name=ModelData.INPUT_NAME+str(i), 
                dtype=ModelData.DTYPE,shape=(1,64,ModelData.H,ModelData.W))
        random_weights=np.random.rand(64, out_maps[i], 3, 3).astype(float_type)-0.5
        conv = network.add_convolution_nd(input=input_tensor, num_output_maps=out_maps[i],
                kernel_shape=(3, 3), kernel=random_weights)
        conv.stride_nd=(1,1)
        conv.padding_nd=(1,1)
        relu = network.add_activation(input=conv.get_output(0), 
                type=trt.ActivationType.RELU)
        relu.get_output(0).name = ModelData.OUTPUT_NAME + str(i)
        network.mark_output(tensor=relu.get_output(0))

def def_group_conv_network(network):
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE,
            shape=(1,320,ModelData.H,ModelData.W))

    random_weights=np.random.rand(64, 15, 3, 3).astype(float_type)-0.5
    grp_conv1 = network.add_convolution_nd(input=input_tensor, num_output_maps=15,
            kernel_shape=(3, 3), kernel=random_weights)
    grp_conv1.stride_nd=(1,1)
    grp_conv1.padding_nd=(1,1)
    grp_conv1.num_groups=5

    relu1 = network.add_activation(input=grp_conv1.get_output(0), 
            type=trt.ActivationType.RELU)

    relu1.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=relu1.get_output(0))

cuda.stop_profiler()
# USE THE ONE YOU WOULD LIKE TO TEST
#def_conv_network(network)
#def_merged_conv_network(network)
#def_multi_conv_network(network)
def_group_conv_network(network)

# BUILD NETWORK
config = builder.create_builder_config()
plan = builder.build_serialized_network(network, config)

# DESERIALIZE NETWORK AND CREATE ENGINE
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(plan)
print('Built engine.')

inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

print('Testing the network...')
time_arr=[]
cuda.start_profiler()
for i in range(101):
    output = do_inference_v2(context, bindings=bindings, inputs=inputs,
            outputs=outputs, stream=stream, time_arr=time_arr)
time_arr = time_arr[1:]
print('Average time:', round(sum(time_arr)/len(time_arr),3), 'ms')
