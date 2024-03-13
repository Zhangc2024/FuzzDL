results = dict()
import tensorflow as tf
try:
  try:
    with tf.device('/CPU'):
      arg_0_0 = 1e+20
      arg_0_1 = -40.0
      arg_0_2 = 70.0
      arg_0_3 = 2.0
      arg_0 = [arg_0_0,arg_0_1,arg_0_2,arg_0_3,]
      arg_1 = tf.int32
      results["res_cpu"] = tf.cast(arg_0,arg_1,)
  except Exception as e:
    results["err_cpu"] = "Error:"+str(e)
  try:
    with tf.device('/GPU:1'):
      arg_0 = [arg_0_0,arg_0_1,arg_0_2,arg_0_3,]
      arg_1 = tf.int32
      results["res_gpu"] = tf.cast(arg_0,arg_1,)
  except Exception as e:
    results["err_gpu"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
