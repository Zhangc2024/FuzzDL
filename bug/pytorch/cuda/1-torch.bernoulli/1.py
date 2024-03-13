results = dict()
import torch
arg_1_tensor = torch.rand([2], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = torch.bernoulli(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.bernoulli(arg_1,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
