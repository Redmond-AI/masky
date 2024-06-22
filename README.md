# Greenscreen Video Processing Script

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
