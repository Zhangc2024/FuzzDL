results = dict()
import torch
arg_1_tensor = torch.rand([1, 4], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-128,32768,[1, 4], dtype=torch.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = "mean"
try:
  results["res_cpu"] = torch.nn.functional.multilabel_margin_loss(arg_1,arg_2,reduction=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.multilabel_margin_loss(arg_1,arg_2,reduction=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
