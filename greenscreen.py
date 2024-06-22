import cv2
import numpy as np
import argparse
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import torch
import os
import subprocess
import shutil
import json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import tifffile as tiff
import traceback

def get_video_bitrate(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'format=bit_rate', '-of', 'json', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    bitrate_info = json.loads(result.stdout)
    return int(bitrate_info['format']['bit_rate'])


def process_images_edgeextend(rgb_image, mask_image, frame_index, debug):

    # Create debug directory if it doesn't exist
    if debug:
        debug_dir = os.path.join(os.path.dirname(__file__), 'debug')
        os.makedirs(debug_dir, exist_ok=True)

    # Ensure mask_image is in grayscale and 8-bit
    if len(mask_image.shape) == 3:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    if mask_image.dtype != np.uint8:
        mask_image = (mask_image * 255).astype(np.uint8)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'01_mask_image_{frame_index:04d}.png'), mask_image)

    # Erode mask_image and normalize
    erosion_size = int(0.004 * mask_image.shape[1])
    if erosion_size % 2 == 0:
        erosion_size += 1
    erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
    mask_image_dilate = cv2.erode(mask_image, erosion_kernel, iterations=1)
    mask_image_dilate = cv2.normalize(mask_image_dilate, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'02-A_mask_image_dilate_{frame_index:04d}.png'), mask_image_dilate)

    # Make a copy of rgb_image named dilate_rgb_image
    dilate_rgb_image = rgb_image.copy()

    # invert the mask
    # invert_mask_image = (255 - mask_image_dilate)

    # Make the pixels in rgb_image black where the pixels in mask_image are black
    dilate_rgb_image[mask_image_dilate < 5] = 0

    # # Expand dimensions of invert_mask_image to match rgb_image
    # invert_mask_image = invert_mask_image[:, :, None]

    # # Make the pixels in rgb_image black where the pixels in mask_image are black
    # dilate_rgb_image = dilate_rgb_image * invert_mask_image

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'02-B_dilate_rgb_image_{frame_index:04d}.png'), dilate_rgb_image)
    
    # Calculate dilation size and ensure it is odd
    dilation_size = int(0.0025 * rgb_image.shape[1])
    if dilation_size % 2 == 0:
        dilation_size += 1
    # Create a circular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    dilate_rgb_image = cv2.dilate(dilate_rgb_image, kernel, iterations=3)
    # cv2.imwrite(os.path.join(debug_dir, f'dilate_rgb_image_01_{frame_index:04d}.png'), dilate_rgb_image)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'03_dilate_rgb_image_{frame_index:04d}.png'), dilate_rgb_image)

    # Blur dilate_rgb_image by 2% of the width
    blur_size = int(0.02 * mask_image.shape[1])
    if blur_size % 2 == 0:
        blur_size += 1
    #dilate_rgb_image = cv2.GaussianBlur(dilate_rgb_image, (blur_size, blur_size), 0)

    #dilate_rgb_image = cv2.dilate(dilate_rgb_image, kernel, iterations=1)

    # cv2.imwrite(os.path.join(debug_dir, f'dilate_rgb_image_02_{frame_index:04d}.png'), dilate_rgb_image)

    # Do edge detection on mask_image and name it edges_mask
    edges_mask = cv2.Canny(mask_image, 100, 200)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'04_edges_mask_01_{frame_index:04d}.png'), edges_mask)

    # Calculate edge dilation size and ensure it is odd
    edge_dilation_size = int(0.002 * mask_image.shape[1])
    if edge_dilation_size % 2 == 0:
        edge_dilation_size += 1
    edge_kernel = np.ones((edge_dilation_size, edge_dilation_size), np.uint8)
    edges_mask = cv2.dilate(edges_mask, edge_kernel, iterations=1)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'05_edges_mask_02_{frame_index:04d}.png'), edges_mask)

    # Blur edges_mask by 2% of the width
    blur_size = int(0.01 * mask_image.shape[1])
    if blur_size % 2 == 0:
        blur_size += 1
    edges_mask = cv2.GaussianBlur(edges_mask, (blur_size, blur_size), 0)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'06_edges_mask_03_{frame_index:04d}.png'), edges_mask)

    # Blend dilate_rgb_image on top of rgb_image and use edges_mask as the mask
    edges_mask_normalized = edges_mask / 255.0
    edge_extend_rgb = rgb_image * (1 - edges_mask_normalized[:, :, None]) + dilate_rgb_image * edges_mask_normalized[:, :, None]

    # Convert edge_extend_rgb to uint8
    edge_extend_rgb = edge_extend_rgb.astype(np.uint8)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, f'07_edge_extend_rgb{frame_index:04d}.png'), edge_extend_rgb)


    # Save images to debug directory
    # cv2.imwrite(os.path.join(debug_dir, f'edges_mask_normalized_{frame_index:04d}.png'), (edges_mask_normalized * 255).astype(np.uint8))
    # cv2.imwrite(os.path.join(debug_dir, f'edge_extend_rgb_{frame_index:04d}.png'), edge_extend_rgb)

    return edge_extend_rgb

def extract_uv_map(uv_map_rgb):
    u_map = uv_map_rgb[0, :, :].clone().detach().cpu().numpy()
    v_map = uv_map_rgb[1, :, :].clone().detach().cpu().numpy()
    return u_map, v_map

def apply_uv_map(frame, u_map, v_map):
    H, W = frame.shape[:2]
    
    u_map = np.clip(u_map, 0, 1)
    v_map = np.clip(v_map, 0, 1)
    
    u_map = u_map * (W - 1)
    v_map = v_map * (H - 1)
    
    u_map = np.clip(u_map, 0, W - 1).astype(np.float32)
    v_map = np.clip(v_map, 0, H - 1).astype(np.float32)

    map_x = u_map.astype(np.float32)
    map_y = v_map.astype(np.float32)

    frame_flipped = np.flipud(frame)
    
    distorted_frame = cv2.remap(frame_flipped, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    distorted_frame = np.clip(distorted_frame, 0, 255).astype(np.uint8)
    
    return distorted_frame

def process_frame(queue, frame_width, frame_height, split, blackwhite, png_dir, pbar, blur, scale, edgeextend, debug, distort, u_map, v_map):
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        frame_index, rgb_frame, mask_frame = item

        try:

            if split:
                # Crop the RGB frame to its left half
                rgb_frame = rgb_frame[:, :rgb_frame.shape[1] // 2]
                if debug:
                    cv2.imwrite(os.path.join(debug_dir, f'split_rgb{frame_index:04d}.png'), rgb_frame)

            # Apply distortion if specified
            if distort == "rgb":
                rgb_frame = apply_uv_map(rgb_frame, u_map, v_map)
                if debug:
                    cv2.imwrite(os.path.join(debug_dir, f'distorted_rgb{frame_index:04d}.png'), rgb_frame)

            elif distort == "mask":
                mask_frame = apply_uv_map(mask_frame, u_map, v_map)
                if debug:
                    cv2.imwrite(os.path.join(debug_dir, f'distorted_mask{frame_index:04d}.png'), mask_frame)
            mask_frame = cv2.resize(mask_frame, (rgb_frame.shape[1], rgb_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            if edgeextend:
                rgb_frame = process_images_edgeextend(rgb_frame, mask_frame, frame_index, debug)


            # Scale up rgb_frame and mask_frame
            if scale != 1.0:
                rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                mask_frame = cv2.resize(mask_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # Ensure the mask frame is in grayscale
            if len(mask_frame.shape) == 3:
                mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

            # Normalize mask frame to range [0, 1]
            mask_frame_normalized = mask_frame / 255.0
            mask_frame_normalized = 1 - mask_frame_normalized

            if blackwhite:
                # Apply threshold to make mask_frame_normalized binary
                mask_frame_normalized = np.where(mask_frame_normalized >= 0.5, 1, 0)

            if blur > 0:
                # Ensure blur is a positive odd number
                if blur % 2 == 0:
                    blur += 1
                # Ensure mask_frame_normalized is float32
                mask_frame_normalized = mask_frame_normalized.astype(np.float32)
                mask_frame_normalized = cv2.GaussianBlur(mask_frame_normalized, (blur, blur), 0)


            # Create a green overlay
            green_overlay = np.zeros_like(rgb_frame)
            # green_overlay[:, :, 1] = 255  # Set green channel to 255

            # NA specific color
            green_overlay[:, :, 0] = int(0.23922 * 255)  # Set red channel
            green_overlay[:, :, 1] = int(1 * 255)        # Set green channel
            green_overlay[:, :, 2] = int(0.00392 * 255)  # Set blue channel
            # Blend the RGB frame with the green overlay based on the mask
            blended_frame = rgb_frame * (1 - mask_frame_normalized[:, :, None]) + green_overlay * mask_frame_normalized[:, :, None]

            # Convert blended_frame to uint8
            blended_frame = blended_frame.astype(np.uint8)

            # Write the frame to a PNG image
            cv2.imwrite(os.path.join(png_dir, f'frame_{frame_index:04d}.png'), blended_frame)

            pbar.update(1)
            # print(f"Processed frame {frame_index}")
        except Exception as e:

            print(f"Error processing frame {frame_index}: {e}")
            traceback.print_exc()
        finally:
            queue.task_done()

def process_videos(rgb_video_path, mask_video_path, split, blackwhite, blur, output_path, scale, edgeextend, debug, distort, distort_tiff):



    # Open the video files
    rgb_cap = cv2.VideoCapture(rgb_video_path)
    mask_cap = cv2.VideoCapture(mask_video_path)

    # Get the properties of the RGB video
    frame_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = rgb_cap.get(cv2.CAP_PROP_FPS)
    print("frames per second", fps)
    frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the bitrate of the RGB video
    bitrate = get_video_bitrate(rgb_video_path)
    print("bitrate", bitrate)
    # bitrate = int(bitrate / 2)

    # Create a directory to store the PNG images
    png_dir = os.path.join(os.path.dirname(rgb_video_path), 'png_images')
    try:
        shutil.rmtree(png_dir)
    except Exception:
        pass
    os.makedirs(png_dir, exist_ok=True)

    # Load UV map if distortion is specified
    u_map, v_map = None, None
    if distort and distort_tiff:
        uv_map_rgb = tiff.imread(distort_tiff).astype(np.float32)
        uv_map_rgb = torch.from_numpy(uv_map_rgb).permute(2, 0, 1)

        u_map, v_map = extract_uv_map(uv_map_rgb)

    # Create a progress bar
    pbar = tqdm(total=frame_count)

    # Create a queue to hold frames
    queue = Queue(maxsize=16)  # Increased maxsize to allow some buffering

    # Start worker threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_frame, queue, frame_width, frame_height, split, blackwhite, png_dir, pbar, blur, scale, edgeextend, debug, distort, u_map, v_map) for _ in range(8)]

        frame_index = 0
        while rgb_cap.isOpened() and mask_cap.isOpened():
            ret_rgb, rgb_frame = rgb_cap.read()
            ret_mask, mask_frame = mask_cap.read()

            if not ret_rgb or not ret_mask:
                break

            queue.put((frame_index, rgb_frame, mask_frame))
            # print(f"Added frame {frame_index} to queue")
            frame_index += 1

        # Stop workers
        for _ in range(8):
            queue.put(None)
            # print("Sent stop signal to worker")

        # Ensure all frames are processed
        queue.join()

        # Wait for all threads to complete
        for future in futures:
            future.result()

    # Close the progress bar
    pbar.close()

    # Release everything
    rgb_cap.release()
    mask_cap.release()

    # Use ffmpeg to encode the output video from the PNG images
    subprocess.run([
        'ffmpeg',
        '-r', str(fps),  # Set the frame rate
        '-i', os.path.join(png_dir, 'frame_%04d.png'),  # Input PNG images
        '-vcodec', 'libx264',  # Set the video codec
        '-b:v', '31000000',  # Set the bitrate to 31.00 Mbit/s
        '-pix_fmt', 'yuv420p',  # Set the pixel format
        '-y',  # Overwrite output file if it exists
        output_path  # Output file
    ])

    # Extract audio from the original video
    audio_path = rgb_video_path.rsplit('.', 1)[0] + '_audio.aac'
    subprocess.run([
        'ffmpeg',
        '-i', rgb_video_path,  # Input video
        '-q:a', '0',  # Best quality
        '-map', 'a',  # Select audio stream
        '-y',  # Overwrite output file if it exists
        audio_path  # Output audio file
    ])

    # Merge the extracted audio with the output video
    output_path_with_audio = output_path.rsplit('.', 1)[0] + '_with_audio.mp4'
    subprocess.run([
        'ffmpeg',
        '-i', output_path,  # Input video
        '-i', audio_path,  # Input audio
        '-c:v', 'copy',  # Copy video codec
        '-c:a', 'aac',  # Set audio codec
        '-strict', 'experimental',  # Allow experimental codecs
        '-y',  # Overwrite output file if it exists
        output_path_with_audio  # Output file with audio
    ])

    # Clean up temporary files
    os.remove(audio_path)
    shutil.rmtree(png_dir)

    print(f"Processing complete. Output saved to {output_path_with_audio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RGB and mask videos to apply a green screen effect.")
    parser.add_argument("--rgb_video_path", type=str, help="Path to the RGB video file.")
    parser.add_argument("--mask_video_path", type=str, help="Path to the mask video file.")
    parser.add_argument("--split", action="store_true", help="Split the RGB frame and resize the mask frame accordingly.")
    parser.add_argument("--blackwhite", action="store_true", help="Process mask_frame_normalized to be binary (0 or 1).")
    parser.add_argument("--blur", type=int, default=0, help="Apply Gaussian blur with the specified kernel size.")
    parser.add_argument("--output", type=str, help="Path to the output video file.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the RGB frame.")
    parser.add_argument("--edgeextend", action="store_true", help="Apply edge extension to the RGB frame.")
    parser.add_argument("--debug", action="store_true", help="Save debug images to the debug directory.")
    parser.add_argument("--distort", type=str, choices=["rgb", "mask"], help="Apply distortion to RGB or mask frame")
    parser.add_argument("--distorttiff", type=str, help="Path to the UV map TIFF file for distortion")

    args = parser.parse_args()

    # Create debug directory if it doesn't exist
    debug_dir = os.path.join(os.path.dirname(__file__), 'debug')

    try:
        shutil.rmtree(debug_dir)
    except Exception:
        pass

    process_videos(args.rgb_video_path, args.mask_video_path, args.split, args.blackwhite, args.blur, args.output, args.scale, args.edgeextend, args.debug, args.distort, args.distorttiff)
