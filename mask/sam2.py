from transformers import Sam2VideoModel, Sam2VideoProcessor, infer_device
import torch, json
import numpy as np


def get_masks(video_frames, model_name="facebook/sam2.1-hiera-tiny", annotations_filepath="annotations.json", save_path="IO/masks.pt"):
    """
    Input video_frames: numpy array [T, H, W, C]
    Output masks_path: path to saved masks tensor [T, H, W]
    """

    device = infer_device()
    model = Sam2VideoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(model_name)

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )

    with open(annotations_filepath, "r") as f:
        clicked_points = json.load(f)["annotations"]
   
    # Add inputs for all frames, even if no annotations exist
    num_frames = video_frames.shape[0]
    for frame_idx in range(num_frames):
        ann_frame_idx_str = str(frame_idx)
        if clicked_points[ann_frame_idx_str] != {}:
            annotations = clicked_points[ann_frame_idx_str]
            obj_ids = []
            input_points = []
            input_labels = []
            for ann_obj_id, points in annotations.items():
                obj_ids.append(int(ann_obj_id))
                input_points.append(points)
                input_labels.append([1]*len(points))

            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_ids=obj_ids,
                input_points=[input_points],
                input_labels=[input_labels]
            )

    # Segment the object on the first frame
    outputs = model(
        inference_session=inference_session,
        frame_idx=0,
    )

    # Get the object IDs and masks from the first frame's segmentation
    initial_obj_ids = inference_session.obj_ids
    initial_masks = processor.post_process_masks(
        [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]

     # Propagate through the entire video starting from the first frame's results
    video_segments = {}
    for sam2_video_output in model.propagate_in_video_iterator(
        inference_session):
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
        )[0]
        video_segments[sam2_video_output.frame_idx] = {
            obj_id: video_res_masks[i]
            for i, obj_id in enumerate(inference_session.obj_ids)
        }


    print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")

    mask = np.zeros((len(video_segments), inference_session.video_height, inference_session.video_width), dtype=np.uint8)

    #Convert to pytorch tensor mask [1, T, H, W]
    for frame_idx in video_segments.keys():
        for _, obj_mask in video_segments[frame_idx].items():

            obj_mask = obj_mask.to(torch.float32).detach().cpu().numpy()
            obj_mask = (obj_mask > 0.5).astype(np.uint8)
            mask[frame_idx] = np.maximum(mask[frame_idx], obj_mask)

    mask = torch.from_numpy(mask).float()  # [T, H, W]

    torch.save(mask, save_path)



# print(len(video_segments))          # → 180 frames
# print(video_segments[0].keys())     # → dict_keys([2, 3])
# print(video_segments[0][2].shape)   # → torch.Size([1, 480, 854])
# print(video_segments[0][2].dtype)   # → torch.float32