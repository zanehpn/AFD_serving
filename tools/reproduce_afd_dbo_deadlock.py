"""Two-rank reproduction of AFD DBO send/send/recv/recv ordering."""
import os,sys,time,threading
import torch
from vllm.distributed.utils import StatelessProcessGroup
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
rank=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank)
fixed=sys.argv[1] in ('fixed', 'patched')
base=int(os.environ.get('REPRO_PORT','29720'))
a=PyNcclCommunicator(StatelessProcessGroup.create('127.0.0.1',base,rank,2),device=rank)
e=PyNcclCommunicator(StatelessProcessGroup.create('127.0.0.1',base+1,rank,2),device=rank)
a.disabled=False;e.disabled=False
x=[torch.full((1536,2048),float(i+1),dtype=torch.bfloat16,device=rank) for i in range(2)]
y=[torch.empty_like(t) for t in x]
compute=torch.cuda.current_stream(); send=torch.cuda.Stream() if fixed else compute
# Warm copies and pointwise kernels before the bounded protocol check.
for i in range(2):y[i].copy_(x[i]);y[i].add_(1)
torch.cuda.synchronize()
print('PROTOCOL_START',rank,'fixed',fixed,flush=True)
threading.Timer(20,lambda:os._exit(124)).start()
if rank==0:
 if sys.argv[1]=='patched':
  from types import SimpleNamespace as NS
  from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector
  connector=object.__new__(P2pNcclAFDConnector)
  connector.vllm_config=NS(model_config=NS(enforce_eager=True))
  connector._send_attn_output_on_current_stream=lambda t,c,**kw:a.send(t,1,stream=torch.cuda.current_stream())
  connector.e2a_group=None; connector.e2a_comm_id=None; connector.tensor_metadata_list=[None,None]
  def receive(*args,ref_tensor):
   e.recv(ref_tensor,1,stream=torch.cuda.current_stream()); return ref_tensor
  connector._recv_hidden_states=receive
  for i in range(2):connector.send_attn_output(x[i],NS(metadata=NS(stage_idx=i)))
  for i in range(2):y[i]=connector.recv_ffn_output(x[i],ubatch_idx=i)
 else:
  events=[]
  for i in range(2):
   send.wait_stream(compute)
   with torch.cuda.stream(send):
    a.send(x[i],1,stream=send); events.append(send.record_event())
  for i in range(2):
   compute.wait_event(events[i]);e.recv(y[i],1,stream=compute)
else:
 for i in range(2):
  a.recv(x[i],0,stream=compute)
  y[i].copy_(x[i]); y[i].add_(1)
  e.send(y[i],0,stream=compute)
torch.cuda.synchronize()
assert y[0][0,0].item()==2 and y[1][0,0].item()==3
print('PROTOCOL_PASS',rank,flush=True)
sys.stdout.flush(); os._exit(0)
