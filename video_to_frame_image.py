import os
import cv2


def video_to_frames(video_file_path, output_dir):
    # Create directory to store output frames
    os.makedirs(output_dir, exist_ok=True)

    # Read video file
    video_capture = cv2.VideoCapture(video_file_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through each frame and save as PNG
    for i in range(frame_count):
        success, frame = video_capture.read()
        if not success:
            break
        frame_path = os.path.join(output_dir, f"frame_{i:05d}.png")
        cv2.imwrite(frame_path, frame)

    # Release video capture object
    video_capture.release()

# Example usage


file_name = 'end_turn_6.mov'
video_file_path = f"/Users/shahafbenshushan/PycharmProjects/pythonProject2/Marvel_Sanp/video_files/{file_name}"
print(os.path.splitext(video_file_path)[0])
output_path = f'data/{file_name[:-4]}/'
video_to_frames(video_file_path, output_path)
