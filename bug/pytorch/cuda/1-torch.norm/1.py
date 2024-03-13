results = dict()
import torch
arg_1_tensor = torch.rand([1024, 16], dtype=torch.float16)
arg_1 = arg_1_tensor.clone()
arg_2 = 2.0
try:
  results["res_cpu"] = torch.norm(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.norm(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
