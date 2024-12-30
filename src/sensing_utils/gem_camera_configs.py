import time

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
TOPICS_DEPTH, TOPICS_RGB, RGB_FRAMES, TOPICS_CAM_INFO = dict(), dict(), dict(), dict()
for k in CAM_INDEX_MAP.keys():
    TOPICS_DEPTH[k] = f"/{k}/aligned_depth_to_color/image_raw"
    TOPICS_RGB[k] = f"/{k}/color/image_raw"
    RGB_FRAMES[k] = f"{k}_color_optical_frame"
    TOPICS_CAM_INFO[k] = f"/{k}/color/camera_info"

time.sleep(0.5)