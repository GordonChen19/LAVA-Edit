import torch

from torchvision.transforms import functional as TF


def resize_video_frames(video_frames: torch.Tensor, size: tuple[int, int] = (480, 854)) -> torch.Tensor:
    """
    Input video_frames: [T, H, W, C], uint8 tensor in [0, 255]
    Output video_tensor: [1, C, T, crop_h, crop_w], float32 
    """


    if video_frames.shape[1] > size[0] or video_frames.shape[2] > size[1]:
        cropped_frames = []
        for frame in video_frames:

            frame = frame.permute(2,0,1) # [C, H, W]
            frame = TF.center_crop(frame, size)  # [C, crop_h, crop_w]
            frame = frame.permute(1,2,0)  # [crop_h, crop_w, C]
            cropped_frames.append(frame)
        video_frames = torch.stack(cropped_frames, dim=0)  # [T, crop_h, crop_w, C]
    
    return video_frames
    


def normalize_video_frames(
    video_frames: torch.Tensor,
    size: tuple[int, int] = (480, 854),
) -> torch.Tensor:
    """
    Input video_frames: [T, H, W, C], uint8 tensor in [0, 255]
    Output video_tensor: [1, C, T, crop_h, crop_w], float32 tensor in [-1, 1]
    """
    
    video_frames = video_frames.to(torch.float32)/255.0  
    video_frames = (video_frames - 0.5) / 0.5  # normalize to [-1, 1]
    video_tensor = video_frames.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # [B, C, T, crop_h, crop_w]

    print("Preprocessed video tensor shape:", video_tensor.shape, "in the range [{:.2f}, {:.2f}]".format(video_tensor.min().item(), video_tensor.max().item()))
    return video_tensor

