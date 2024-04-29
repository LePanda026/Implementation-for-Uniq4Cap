from clip_vit import VisionTransformer3D
import torch
from torch import nn


model = VisionTransformer3D(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=23,
        heads=16,
        use_grad_checkpointing=False,
        num_frames=8
    )

# incompatible_keys = model.load_state_dict(state_dict, strict=False)
state_dict = torch.load('/home/knight/LAVIS/mm_llm/ckpt/encoder_ckpt/vit3d.pth', map_location="cpu")
model.load_state_dict(state_dict)
test_tensor = torch.randn((5,3,8,224,224))
output = model(test_tensor)
print()


model.expand3d()
# print(model)
torch.save(model.state_dict(),'/home/knight/LAVIS/mm_llm/ckpt/encoder_ckpt/vit3d.pth')
conv1d = nn.Conv2d(in_channels=3, out_channels=1024, kernel_size=14, stride=14, bias=False)
conv1d.load_state_dict(state_dict,strict=False)
state_dict_expand = conv1d.state_dict()['weight'].unsqueeze(2)
device, dtype = state_dict_expand.device,state_dict_expand.dtype


conv3d = nn.Conv3d(in_channels=3,out_channels=1024,kernel_size=(1,14,14),stride=(1,14,14),bias=False)
conv3d.load_state_dict({'weight':state_dict_expand})


print(state_dict.keys())

print()
