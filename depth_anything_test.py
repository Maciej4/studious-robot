from transformers import pipeline
from PIL import Image
import cv2
import numpy as np
import os
import glob


class DepthEstimator:
    def __init__(self):
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda")

    def estimate_depth(self, image_path: str) -> np.ndarray:
        """
        Take the image located at the image path and generate a depth map from it.
        """
        image = Image.open(image_path)
        depth = self.pipe(image)["depth"]
        depth_array = np.array(depth)
        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        return depth_uint8

    def get_depth_at_fractional_point(self, depth_array: np.ndarray, x: float, y: float) -> float:
        """
        Given a depth map and a fractional point (a point with x and y coordinates between 0 and 1),
        this function returns the depth at that fractional point.
        """
        height, width = depth_array.shape
        x = int(x * width)
        y = int(y * height)
        return (255 - int(depth_array[y, x])) / 10


def main():
    depth_estimator = DepthEstimator()

    # Get the latest Minecraft screenshot
    # The path is to the Minecraft directory on Windows when accessing from within WSL. This
    # path will need to be changed.
    list_of_screenshots = glob.glob('/mnt/c/Users/m/AppData/Roaming/.minecraft/screenshots/*.png')
    latest_screenshot = max(list_of_screenshots, key=os.path.getctime)

    # Estimate the depth
    depth = depth_estimator.estimate_depth(latest_screenshot)

    # Write the depth data to a file
    cv2.imwrite("depth.png", depth)

    print(depth_estimator.get_depth_at_fractional_point(depth, 0.5, 0.5))


if __name__ == "__main__":
    main()
