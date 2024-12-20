BASE_FRAME = 'fr3_link0'
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640

INHAND_CAMERA_NAME = 'tim'
CAM_INDEX_MAP = {
            'dave': 0,
        }
CAM_INDEX_MAP = {k: v for k, v in sorted(CAM_INDEX_MAP.items(), key=lambda item: item[1])}
# Initialize depth topics and subscriptions
TOPICS_DEPTH = {
    'dave': "/dave/aligned_depth_to_color/image_raw",
}
TOPICS_RGB = {
    'dave': "/dave/color/image_raw",
}
RGB_FRAMES = {
    'dave': "dave_color_optical_frame",
}
TOPICS_CAM_INFO = {
    'dave': "/dave/color/camera_info",
}