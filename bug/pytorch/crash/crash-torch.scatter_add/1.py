results = dict()
import torch
arg_1_tensor = torch.rand([10, 5], dtype=torch.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3_tensor = torch.randint(-2048,4096,[5, 5], dtype=torch.int64)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = torch.rand([5, 5], dtype=torch.float64)
arg_4 = arg_4_tensor.clone()
try:
  results["res_cpu"] = torch.scatter_add(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.scatter_add(arg_1,arg_2,arg_3,arg_4,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
