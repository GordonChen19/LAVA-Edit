import os, json, argparse, torch, subprocess
from torchvision.io import write_video
from transformers.video_utils import load_video
from mask.view import annotate
from mask.sam2 import get_masks
from mask.prepare_video import resize_video_frames


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
        "--edit_prompt",
        type=str,
        default="Two kids jump off the bed",
        help="Prompt describing the desired edit"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["annotate", "prepare_masks", "train", "inference"],
        default ="train",
        help="Stage of the video editing pipeline",
    )
    parser.add_argument(
        "--annotations_save_path",
        type=str,
        default="mask/annotations.json",
        help="Path to save the annotations",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
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
    parser.add_argument(
        "--current_dir",
        type=str,
        default="/projects_vol/gp_slab/chen2008/lava",
        help="Current working directory",
    )

    args = parser.parse_args()

    video_path = args.video_path
    stage = args.stage
    fps = args.fps
    seconds = args.seconds
    identifier = args.identifier
    video_caption = args.video_caption
    edit_prompt = args.edit_prompt
    current_dir = args.current_dir
    annotations_save_path = os.path.join(current_dir, args.annotations_save_path)

    video_frames, orig_fps = load_video(video_path)
    video_frames = video_frames[:fps*seconds]  #Take first n seconds of the video [T, H, W, C]
    video_frames = resize_video_frames(torch.tensor(video_frames))

    IO_save_path = os.path.join(current_dir, "IO")
    video_folder = os.path.join(IO_save_path,"inputs")
    mask_folder = os.path.join(IO_save_path,"masks")


    os.makedirs(IO_save_path, exist_ok = True)
    os.makedirs(video_folder, exist_ok = True)
    os.makedirs(mask_folder, exist_ok = True)
    write_video(os.path.join(video_folder, f"{identifier}.mp4"), video_frames, fps=fps)
    
    with open(os.path.join(video_folder, f"{identifier}.txt"), "w") as f:
        f.write(video_caption)

    
    
    if stage == "annotate":
        #Done with CPU
        annotations = annotate(video_frames)
        
        with open(annotations_save_path, "w") as f:
            json.dump({"annotations": annotations}, f)

    elif stage == "prepare_masks":
        
        get_masks(video_frames, 
                  annotations_filepath=annotations_save_path, 
                  save_path= os.path.join(mask_folder, f"{identifier}.pt"))

    elif stage == "train":
        # Set environment variables for single-GPU DeepSpeed training

        subprocess.run(
            [
                "deepspeed",
                "--num_gpus=1",
                os.path.join(current_dir, "diffusion-pipe/train.py"),
                "--deepspeed",
                "--config",
                os.path.join(current_dir, "diffusion-pipe/examples/wan_i2v.toml"),
            ]
        )
        
        
    elif stage == "inference":
        pass
     

#For Annotation, run:
# python main.py --video_path path/to/video.mp4 --stage annotate --annotations_save_path


# pip install "huggingface_hub[cli]"
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./Wan2.1-I2V-14B-480P


# /Users/gordonchen/Library/Mobile Documents/com~apple~CloudDocs/Papers/Video Editing/src/mask