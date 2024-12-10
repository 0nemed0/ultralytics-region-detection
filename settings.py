# ORIGINAL file was from Rampal Punia

from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())


# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / '01_output_video.mp4',
    #'video_2': VIDEO_DIR / 'video_2.mp4',
    #'video_3': VIDEO_DIR / 'video_3.mp4',
}


# Webcam
WEBCAM_PATH = 0
