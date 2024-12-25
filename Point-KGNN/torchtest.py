
import torch
import torchvision
print('torch版本号:',torch.__version__,'cuda是否可用:',torch.cuda.is_available())
print('torchvision版本号:',torchvision.__version__)
print('一共{}块gpu设备'.format(torch.cuda.device_count()))
print('第一块gpu设备名:{}'.format(torch.cuda.get_device_name(0)))
print('cuda版本号:',torch.version.cuda)
print('cudnn版本号:',torch.backends.cudnn.version())
print('cuda计算测试:',torch.rand(3,3).cuda())