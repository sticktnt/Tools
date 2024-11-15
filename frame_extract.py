import os
from multiprocessing import Pool, cpu_count, RLock

import cv2

from tqdm import tqdm


def frame_extract(video_path, extract_frame_size, output_dir_path, position=None):
    """
    Extract frames from a video file.

    :param video_path: Path to the video file.
    :param extract_frame_size: Number of frames to extract.
    :param output_dir_path: Path to the output directory.
    :param position: progress bar position
    :return: None
    """
    if not os.path.exists(video_path):
        print(f"{video_path} not exist.")
        return 0

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"create {output_dir_path} successfully.")

    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = frame_total // extract_frame_size

    pbar = tqdm(total=frame_total, desc=f"Extracting frames from {video_name}", position=position)

    i = 0
    while True:
        ref, frame = cap.read()
        if not ref:
            break
        if i % interval == 0:
            cv2.imwrite(os.path.join(output_dir_path, f"{video_name}_{i}.jpg"), frame)
        i += 1
        pbar.update(1)
    pbar.close()
    cap.release()


def frame_extract_from_dir_with_mul_thread(videos_dir_path, extract_frame_size, output_dir_path,
                                           process_count=cpu_count):
    """
    Extract frames from all video files in a directory.

    :param videos_dir_path: Path to the directory containing video files.
    :param extract_frame_size: Number of frames to extract.
    :param output_dir_path: Path to the output directory.
    :param process_count: process numbers
    :return: None
    """
    # Get the list of video files
    video_files = [f for f in os.listdir(videos_dir_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Prepare arguments for each video file
    args = [
        (os.path.join(videos_dir_path, video), extract_frame_size, os.path.join(output_dir_path, video.split('.')[0]),
         index)
        for index, video in enumerate(video_files)]

    # Use a pool of workers to process videos in parallel
    with Pool(processes=process_count) as pool:
        pool.starmap(frame_extract, args)


if __name__ == "__main__":
    frame_extract_from_dir_with_mul_thread("./test_videos", 1, "./result", 6)
