Links to download UVmaps:
https://www.dropbox.com/scl/fo/2la67wc5noe0qh855xd77/AJICRux2JReKDmCiWAu4FZU?rlkey=yu78iwgj2bi3pkh1or3si1vwh&dl=0



# Greenscreen Video Processing Script Greenscreen.py

This Python script processes RGB and mask videos to apply a greenscreen effect. It takes an RGB video and a corresponding mask video as input, and produces an output video where the masked areas are replaced with a green background.

## Features

- Applies greenscreen effect based on a mask video
- Supports various processing options like splitting frames, edge extension, and distortion
- Can apply blur to the mask
- Allows scaling of the output video
- Supports debug mode for saving intermediate processing steps
- Preserves audio from the original RGB video

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- tqdm
- moviepy
- PyTorch
- tifffile

You also need to have `ffmpeg` installed and accessible from the command line.

## Usage

```
python script_name.py --rgb_video_path PATH --mask_video_path PATH --output PATH [OPTIONS]
```

### Required Arguments

- `--rgb_video_path`: Path to the input RGB video file.
- `--mask_video_path`: Path to the input mask video file.
- `--output`: Path where the output video file will be saved.

### Optional Arguments

- `--split`: If set, splits the RGB frame in half and resizes the mask frame accordingly. This is useful when the RGB and mask are side-by-side in the same video.
- `--blackwhite`: If set, processes the mask to be binary (either 0 or 1). This creates a hard edge in the mask.
- `--blur BLUR`: Applies Gaussian blur to the mask with the specified kernel size. Must be an odd number. Higher values create a softer edge.
- `--scale SCALE`: Scales the output video by the specified factor. Default is 1.0 (no scaling).
- `--edgeextend`: Applies an edge extension algorithm to the RGB frame. This can help reduce artifacts around the edges of the mask.
- `--debug`: Saves intermediate processing steps as images in a 'debug' directory. Useful for troubleshooting.
- `--distort {rgb,mask}`: Applies distortion to either the RGB or mask frame based on a provided UV map.
- `--distorttiff PATH`: Path to a TIFF file containing the UV map for distortion. Required if `--distort` is used.

## How It Works

1. The script reads the RGB and mask videos frame by frame.
2. Each frame is processed according to the specified options:
   - If `--split` is used, the RGB frame is cropped to its left half.
   - If `--distort` is used, the specified frame (RGB or mask) is distorted using the provided UV map.
   - If `--edgeextend` is used, an edge extension algorithm is applied to reduce artifacts.
   - The mask is normalized and optionally binarized if `--blackwhite` is used.
   - If `--blur` is specified, Gaussian blur is applied to the mask.
   - The RGB frame is blended with a green background based on the processed mask.
3. Processed frames are saved as PNG images.
4. `ffmpeg` is used to encode the PNG images into a video.
5. Audio from the original RGB video is extracted and merged with the new video.

## Performance Considerations

- The script uses multi-threading to process frames in parallel, which can significantly speed up processing for longer videos.
- Output video bitrate is set to 31 Mbit/s, which can be adjusted in the code if needed.
- Temporary PNG files and audio are cleaned up after processing.

## Debugging

If you encounter issues or want to inspect the intermediate steps:
1. Run the script with the `--debug` flag.
2. Check the 'debug' directory created in the same location as the script for step-by-step processing images.

## Notes

- Ensure that your RGB and mask videos have the same frame rate and duration.
- The green color used for the background is set to RGB(61, 255, 1), which can be modified in the code if needed.
- The script assumes that in the mask video, black represents the foreground (area to keep) and white represents the background (area to replace with green).

## Example Command

```
python greenscreen_processor.py --rgb_video_path input.mp4 --mask_video_path mask.mp4 --output output.mp4 --split --blur 5 --edgeextend --debug
```

This command will process 'input.mp4' with 'mask.mp4', splitting the input frame, applying a blur of 5 to the mask, using edge extension, and saving debug images. The result will be saved as 'output.mp4'.# masky
Tools for distorting and undistorting videos and using masks to create greenscreen videos


# ________________________________________________________________________________________________


# UV Map Video Distortion Script undistort_02.py

This Python script applies UV map distortion to an input video based on a provided TIFF UV map. It processes the video frame by frame, applies the distortion, and creates a new output video.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- PyTorch
- tifffile
- tqdm
- ffmpeg (command-line tool)

## Usage

```
python script_name.py --input_video INPUT_VIDEO --output_video OUTPUT_VIDEO --uv_map UV_MAP [--split] [--num_workers NUM_WORKERS] [--temp_folder TEMP_FOLDER] [--blur BLUR] [--max_res MAX_RES]
```

## Arguments

1. `--input_video`: Path to the input video file.
   - This is the RGB video you want to distort.

2. `--output_video`: Path to the output video file.
   - This is where the distorted video will be saved.

3. `--uv_map`: Path to the UV map TIFF file.
   - This TIFF file contains the UV map used for distorting the video.

4. `--split` (optional): Flag to split the video in half.
   - If set, only the left half of each frame will be processed.
   - Default: False

5. `--num_workers` (optional): Number of worker threads for parallel processing.
   - This determines how many frames are processed simultaneously.
   - Default: 4

6. `--temp_folder` (optional): Temporary folder for intermediate files.
   - This folder will be used to store individual processed frames before combining them into a video.
   - Default: 'temp'

7. `--blur` (optional): Gaussian blur kernel size.
   - If set to a value greater than 0, a Gaussian blur will be applied to the output frames.
   - Default: 0 (no blur)

8. `--max_res` (optional): Maximum resolution for output frames.
   - This sets the maximum width and height for the processed frames.
   - Default: 2048

## How It Works

1. **Loading the UV Map**: The script loads the TIFF UV map and extracts the U and V components.

2. **Video Processing**:
   - The input video is read frame by frame.
   - If the `--split` option is used, only the left half of each frame is processed.
   - Each frame is resized to the specified `max_res` x `max_res`.
   - The UV map distortion is applied to each frame using the `apply_uv_map` function.
   - The distorted frame is resized back to a maximum of `max_res` x `max_res` using Lanczos interpolation.
   - If blur is specified, it's applied to the frame.
   - The processed frame is saved as a PNG file in the temporary folder.

3. **Parallel Processing**: The script uses a ThreadPoolExecutor to process multiple frames in parallel, speeding up the operation.

4. **Video Creation**: After all frames are processed, ffmpeg is used to combine the PNG files into a video, using the following settings:
   - Frame rate: 30 fps
   - Codec: libx264
   - Preset: slow
   - CRF: 18
   - Pixel format: yuv420p
   - Output resolution: Scaled to fit within `max_res` x `max_res`, preserving aspect ratio

5. **Cleanup**: The temporary folder and its contents are deleted after the video is created.

## Notes

- The script uses Lanczos interpolation for resizing to maintain high image quality.
- The output video is scaled to fit within the specified `max_res` dimensions while preserving the aspect ratio.
- Make sure you have ffmpeg installed and accessible in your system's PATH for the video creation step.
- The `max_res` parameter allows you to control the maximum resolution of the output video, which can be useful for managing file size and processing time.
