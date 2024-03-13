results = dict()
import tensorflow as tf
try:
  try:
    with tf.device('/CPU'):
      arg_0_0_0_0 = 27.0
      arg_0_0_0_1 = -63.0
      arg_0_0_0_2 = 61.0
      arg_0_0_0 = [arg_0_0_0_0,arg_0_0_0_1,arg_0_0_0_2,]
      arg_0_0_1_0 = -1e+20
      arg_0_0_1_1 = -56.0
      arg_0_0_1_2 = 1e+20
      arg_0_0_1 = [arg_0_0_1_0,arg_0_0_1_1,arg_0_0_1_2,]
      arg_0_0 = [arg_0_0_0,arg_0_0_1,]
      arg_0_1_0_0 = 1e+20
      arg_0_1_0_1 = 19.0
      arg_0_1_0_2 = 2
      arg_0_1_0 = [arg_0_1_0_0,arg_0_1_0_1,arg_0_1_0_2,]
      arg_0_1_1_0 = -5.0
      arg_0_1_1_1 = -4.0
      arg_0_1_1_2 = 19.0
      arg_0_1_1 = [arg_0_1_1_0,arg_0_1_1_1,arg_0_1_1_2,]
      arg_0_1 = [arg_0_1_0,arg_0_1_1,]
      arg_0 = [arg_0_0,arg_0_1,]
      dtype = tf.int32
      saturate = False
      results["res_cpu"] = tf.image.convert_image_dtype(arg_0,dtype=dtype,saturate=saturate,)
  except Exception as e:
    results["err_cpu"] = "Error:"+str(e)
  try:
    with tf.device('/GPU:1'):
      arg_0_0_0 = [arg_0_0_0_0,arg_0_0_0_1,arg_0_0_0_2,]
      arg_0_0_1 = [arg_0_0_1_0,arg_0_0_1_1,arg_0_0_1_2,]
      arg_0_0 = [arg_0_0_0,arg_0_0_1,]
      arg_0_1_0 = [arg_0_1_0_0,arg_0_1_0_1,arg_0_1_0_2,]
      arg_0_1_1 = [arg_0_1_1_0,arg_0_1_1_1,arg_0_1_1_2,]
      arg_0_1 = [arg_0_1_0,arg_0_1_1,]
      arg_0 = [arg_0_0,arg_0_1,]
      dtype = tf.int32
      results["res_gpu"] = tf.image.convert_image_dtype(arg_0,dtype=dtype,saturate=saturate,)
  except Exception as e:
    results["err_gpu"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
