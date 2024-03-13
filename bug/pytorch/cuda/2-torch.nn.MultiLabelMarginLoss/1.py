results = dict()
import torch
arg_class = torch.nn.MultiLabelMarginLoss()
arg_1_0_tensor = torch.rand([1, 4], dtype=torch.float32)
arg_1_0 = arg_1_0_tensor.clone()
arg_1_1_tensor = torch.randint(-2,8192,[1, 4], dtype=torch.int64)
arg_1_1 = arg_1_1_tensor.clone()
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_cpu"] = arg_class(*arg_1)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_1_0 = arg_1_0_tensor.clone().cuda()
arg_1_1 = arg_1_1_tensor.clone().cuda()
arg_1 = [arg_1_0,arg_1_1,]
try:
  results["res_gpu"] = arg_class(*arg_1)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
