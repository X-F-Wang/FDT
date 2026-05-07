import torch
import torch.nn.functional as F
from test_mod import PixelMixer_le

# # 假设输入的特征图和注意力分数图
# batch_size = 2
# channels = 3
# height_q = 16
# width_q = 16
#
# feat_q = torch.randn(batch_size, channels, height_q, width_q)
# attention_map = torch.rand(batch_size, 2, 32, 32)
#
# # 生成采样坐标
# sample_coord = torch.rand(batch_size, height_q, width_q, 2)  # 注意：这里只是随机生成示例坐标，实际应使用根据注意力分数计算得到的坐标
#
# # 使用 F.grid_sample 进行双线性插值采样
# sample_feat_q = F.grid_sample(feat_q, sample_coord.flip(-1), mode='bilinear', align_corners=False)
#
# # 输出结果的形状
# print("Sampled Feature Shape:", sample_feat_q.shape)
PM = PixelMixer_le(8)
print(PM.mask)

