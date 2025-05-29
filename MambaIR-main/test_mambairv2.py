# test_mambairv2.py

from basicsr.archs.mambairv2_arch import MambaIRv2
import torch

# 初始化模型
upscale = 4
model = MambaIRv2(
    upscale=2,
    img_size=64,
    embed_dim=48,
    d_state=8,
    depths=[5, 5, 5, 5],
    num_heads=[4, 4, 4, 4],
    window_size=16,
    inner_rank=32,
    num_tokens=64,
    convffn_kernel_size=5,
    img_range=1.,
    mlp_ratio=1.,
    upsampler='pixelshuffledirect').cuda()

# 打印模型参数数量
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.3fM" % (total / 1e6))
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable_num)

# 测试模型
_input = torch.randn([2, 3, 64, 64]).cuda()
output = model(_input).cuda()
print(output.shape)
