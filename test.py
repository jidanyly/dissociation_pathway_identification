import os
from tensorflow.python.client import device_lib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '99'

print(device_lib.list_local_devices())