import os, json, argparse, torch, subprocess
from torchvision.io import write_video
from transformers.video_utils import load_video
from preprocess.view import annotate
from preprocess.sam2 import get_masks
from preprocess.prepare_video import resize_video_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Editing")

    parser.add_argument(
        "--video_path",
        type=str,
        default = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4",
        help="Path to the video file to annotate",
    )
    parser.add_argument(
        "--video_caption",
        type=str,
        default="Two kids jumping on the bed in a bedroom.",
        help="Prompt describing the video content",
    )
    parser.add_argument(
        "edit_prompt",
        type=str,
        default="Two kids jump off the bed",
        help="Prompt describing the desired edit"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["annotate", "train", "inference"],
        default ="train",
        help="Stage of the video editing pipeline",
    )
    parser.add_argument(
        "--annotations_save_path",
        type=str,
        default="annotations.json",
        help="Path to save the annotations",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Target FPS for video resampling",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=5,
        help="Number of seconds to process from the video",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default="example",
        help="The name of the video input (e.g. example.mp4)",
    )

    args = parser.parse_args()

    video_path = args.video_path
    stage = args.stage
    annotations_save_path = args.annotations_save_path
    fps = args.fps
    seconds = args.seconds
    identifier = args.identifier
    video_caption = args.video_caption
    edit_prompt = args.edit_prompt


    video_frames, orig_fps = load_video(video_path)
    video_frames = video_frames[:fps*seconds]  #Take first n seconds of the video [T, H, W, C]
    video_frames = resize_video_frames(torch.tensor(video_frames))

    IO_save_path = "IO"
    video_folder = "inputs"
    mask_folder = "masks"


    os.makedirs(IO_save_path, exist_ok = True)
    os.makedirs(os.path.join(IO_save_path, video_folder), exist_ok = True)
    os.makedirs(os.path.join(IO_save_path, mask_folder), exist_ok = True)
    write_video(os.path.join(IO_save_path, video_folder, f"{identifier}.mp4"), video_frames, fps=fps)
    
    with open(os.path.join(IO_save_path, video_folder, f"{identifier}.txt"), "w") as f:
        f.write(video_caption)

    
    
    if stage == "annotate":
        
        annotations = annotate(video_frames)
        
        with open(annotations_save_path, "w") as f:
            json.dump({"annotations": annotations}, f)

        # Get Mask via Segmentation Model
        get_masks(video_frames, annotations_filepath=annotations_save_path, save_path=os.path.join(IO_save_path, os.path.join(mask_folder, f"{identifier}.pt")))

    elif stage == "train":
        # Set environment variables for single-GPU DeepSpeed training
        env = os.environ.copy()
        env["NCCL_P2P_DISABLE"] = "1"
        env["NCCL_IB_DISABLE"] = "1"
        
        subprocess.run(
            [
                "deepspeed",
                "--num_gpus=1",
                "diffusion-pipe/train.py",
                "--deepspeed",
                "--config",
                "diffusion-pipe/examples/wan_i2v.toml",
            ],
            env=env
        )
        
        
    # elif stage == "inference":
    #     pass
     

#For Annotation, run:
# python main.py --video_path path/to/video.mp4 --stage annotate --annotations_save_path


# pip install "huggingface_hub[cli]"
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P
