import cv2
import numpy as np
import torch
import os
import tifffile as tiff
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import subprocess
import glob

def extract_uv_map(uv_map_rgb):
    u_map = uv_map_rgb[0, :, :].clone().detach().cpu().numpy()
    v_map = uv_map_rgb[1, :, :].clone().detach().cpu().numpy()
    return u_map, v_map

def apply_uv_map(frame, u_map, v_map):
    H, W = frame.shape[-2:]
    
    u_map = np.clip(u_map, 0, 1)
    v_map = np.clip(v_map, 0, 1)
    
    u_map = u_map * (W - 1)
    v_map = v_map * (H - 1)
    
    u_map = np.clip(u_map, 0, W - 1).astype(np.float32)
    v_map = np.clip(v_map, 0, H - 1).astype(np.float32)

    map_x = u_map.astype(np.float32)
    map_y = v_map.astype(np.float32)

    frame_cpu = frame.permute(1, 2, 0).cpu().numpy()

    frame_flipped = np.flipud(frame_cpu)
    
    distorted_frame = cv2.remap(frame_flipped, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    distorted_frame = np.clip(distorted_frame, 0, 255).astype(np.uint8)
    
    return torch.tensor(distorted_frame).permute(2, 0, 1)

def process_video_frame(frame, u_map, v_map, crop_width, idx, temp_folder, blur, max_res):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    frame_cropped = frame[:, :crop_width]

    frame_resized = cv2.resize(frame_cropped, (max_res, max_res), interpolation=cv2.INTER_LINEAR)

    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float()
    
    distorted_frame = apply_uv_map(frame_tensor, u_map, v_map)
    
    result = distorted_frame.permute(1, 2, 0).byte().cpu().numpy()
    
    # Resize the result to a max of max_res x max_res using Lanczos filter
    H, W, _ = result.shape
    if H > max_res or W > max_res:
        scaling_factor = min(max_res / H, max_res / W)
        new_size = (int(W * scaling_factor), int(H * scaling_factor))
        result = cv2.resize(result, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Apply Gaussian blur if blur value is specified
    if blur > 0:
        result = cv2.GaussianBlur(result, (blur, blur), 0)
    
    frame_path = os.path.join(temp_folder, f"frame_{idx:06d}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

def process_video(video_path, output_path, u_map, v_map, split, num_workers, temp_folder, blur, max_res):
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if split:
        crop_width = width // 2
    else:
        crop_width = width
    
    os.makedirs(temp_folder, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx in range(length):
            ret, frame = cap.read()
            if not ret:
                break
            futures[executor.submit(process_video_frame, frame, u_map, v_map, crop_width, idx, temp_folder, blur, max_res)] = idx

        with tqdm(total=length, desc="Processing frames") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing frame {idx}: {e}")
                pbar.update(1)

    cap.release()

    create_video_from_images(temp_folder, output_path, max_res)
    
    shutil.rmtree(temp_folder)

def create_video_from_images(temp_folder, output_path, max_res):
    image_files = sorted(glob.glob(os.path.join(temp_folder, "*.png")))
    
    if not image_files:
        raise ValueError("No images found in the temporary folder.")
    
    # Create a text file with the list of images
    list_file = os.path.join(temp_folder, "image_list.txt")
    with open(list_file, 'w') as f:
        for image_file in image_files:
            f.write(f"file '{os.path.abspath(image_file)}'\n")
    
    # Use ffmpeg to create the video with scaling and specify the lanczos filter
    ffmpeg_command = [
        'ffmpeg', '-y', '-r', '30', '-f', 'concat', '-safe', '0', '-i', list_file,
        '-vf', f'scale={max_res}:{max_res}:force_original_aspect_ratio=decrease:flags=lanczos',
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', output_path
    ]
    subprocess.run(ffmpeg_command, check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process a video with a UV map.')
    parser.add_argument('--input_video', type=str, help='Path to the input video file')
    parser.add_argument('--output_video', type=str, help='Path to the output video file')
    parser.add_argument('--uv_map', type=str, help='Path to the UV map TIFF file')
    parser.add_argument('--split', action='store_true', help='Split the video in half')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--temp_folder', type=str, default='temp', help='Temporary folder for intermediate files')
    parser.add_argument('--blur', type=int, default=0, help='Gaussian blur kernel size')
    parser.add_argument('--max_res', type=int, default=2048, help='Maximum resolution for output frames')
    args = parser.parse_args()

    # Print the values of all arguments
    print(f"Input Video: {args.input_video}")
    print(f"Output Video: {args.output_video}")
    print(f"UV Map: {args.uv_map}")
    print(f"Split: {args.split}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Temporary Folder: {args.temp_folder}")
    print(f"Blur: {args.blur}")
    print(f"Max Resolution: {args.max_res}")

    uv_map_rgb = tiff.imread(args.uv_map).astype(np.float32)
    uv_map_rgb = torch.from_numpy(uv_map_rgb).permute(2, 0, 1)
    u_map, v_map = extract_uv_map(uv_map_rgb)

    process_video(args.input_video, args.output_video, u_map, v_map, args.split, args.num_workers, args.temp_folder, args.blur, args.max_res)
