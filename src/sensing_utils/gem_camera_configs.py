BASE_FRAME = 'fr3_link0'
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640

INHAND_CAMERA_NAME = 'tim'
CAM_INDEX_MAP = {
            'bob': 0,
            'kevin': 1,
            'mel': 3,
        }
CAM_INDEX_MAP = {k: v for k, v in sorted(CAM_INDEX_MAP.items(), key=lambda item: item[1])}
# Initialize depth topics and subscriptions
TOPICS_DEPTH = {
    'bob': "/bob/aligned_depth_to_color/image_raw",
    'kevin': "/kevin/aligned_depth_to_color/image_raw",
    'mel': "/mel/aligned_depth_to_color/image_raw"
}
TOPICS_RGB = {
    'bob': "/bob/color/image_raw",
    'kevin': "/kevin/color/image_raw",
    'mel': "/mel/color/image_raw"
}
RGB_FRAMES = {
    'bob': "bob_color_optical_frame",
    'kevin': "kevin_color_optical_frame",
    'mel': "mel_color_optical_frame"
}
TOPICS_CAM_INFO = {
    'bob': "/bob/color/camera_info",
    'kevin': "/kevin/color/camera_info",
    'mel': "/mel/color/camera_info"
}