BASE_FRAME = 'fr3_link0'
IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640

INHAND_CAMERA_NAME = 'tim'
CAM_INDEX_MAP = {
            'bob': 0,
            'kevin': 1,
            'stuart': 2,
            'dave': 3,
            'mel': 4,
            'tim': 5, # the in hand cam needs 
        }
CAM_INDEX_MAP = {k: v for k, v in sorted(CAM_INDEX_MAP.items(), key=lambda item: item[1])}
# Initialize depth topics and subscriptions
TOPICS_DEPTH = {
    'bob': "/bob/aligned_depth_to_color/image_raw",
    'kevin': "/kevin/aligned_depth_to_color/image_raw",
    'stuart': "/stuart/aligned_depth_to_color/image_raw",
    'tim': "/tim/aligned_depth_to_color/image_raw",
    'dave': "/dave/aligned_depth_to_color/image_raw",
    'mel': "/mel/aligned_depth_to_color/image_raw"
}
TOPICS_RGB = {
    'bob': "/bob/color/image_raw",
    'kevin': "/kevin/color/image_raw",
    'stuart': "/stuart/color/image_raw",
    'tim': "/tim/color/image_rect_raw",
    'dave': "/dave/color/image_raw",
    'mel': "/mel/color/image_raw"
}
RGB_FRAMES = {
    'bob': "bob_color_optical_frame",
    'kevin': "kevin_color_optical_frame",
    'stuart': "stuart_color_optical_frame",
    'tim': "tim_color_optical_frame",
    'dave': "dave_color_optical_frame",
    'mel': "mel_color_optical_frame"
}
TOPICS_CAM_INFO = {
    'bob': "/bob/color/camera_info",
    'kevin': "/kevin/color/camera_info",
    'stuart': "/stuart/color/camera_info",
    'tim': "/tim/color/camera_info",
    'dave': "/dave/color/camera_info",
    'mel': "/mel/color/camera_info"
}