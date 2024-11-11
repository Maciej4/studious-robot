import cv2
import numpy as np
import threading
import time


class PointTracker:
    """
    Track a point in a video stream using Lucas-Kanade optical flow.

    Initialize with the point to track, then call the get_latest_position method to get the
    latest position of the point. Once done, call the stop method to stop the tracking.
    """

    def __init__(self, point, headless=True):
        self.x_coordinate = point[0]
        self.y_coordinate = point[1]
        self.headless = headless
        self.latest_position = point
        self.running = True

        # Initialize video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.image_size = (640, 480)

        # Initialize a thread for tracking
        self.thread = threading.Thread(target=self._run_tracking)
        self.thread.start()

    def _run_tracking(self):
        # Read the first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read video")
            self.cap.release()
            return

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = np.array([[self.x_coordinate, self.y_coordinate]], dtype=np.float32)

        # Parameters for Lucas-Kanade Optical Flow
        lk_params = dict(winSize=(21, 21),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # ORB detector for recovery
        orb = cv2.ORB_create()
        keypoint = cv2.KeyPoint(self.x_coordinate, self.y_coordinate, 20)
        keypoints_prev = [keypoint]
        _, des_prev = orb.compute(frame_gray, keypoints_prev)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(frame_gray, frame_gray_new, p0, None, **lk_params)

            # Check if the point was found
            if st[0][0] == 1:
                p0 = p1  # Update the point for the next frame
                try:
                    self.latest_position = (int(p1[0][0]), int(p1[0][1]))
                except TypeError:
                    self.running = False

                if not self.headless:
                    cv2.circle(frame, self.latest_position, 5, (0, 255, 0), -1)
            else:
                # TODO: This doesn't really work at all. Look for a better solution.
                # print("Tracking failure detected. Attempting ORB-based reinitialization...")
                keypoints_curr = [cv2.KeyPoint(self.latest_position[0], self.latest_position[1], 20)]
                _, des_curr = orb.compute(frame_gray_new, keypoints_curr)

                if des_prev is not None and des_curr is not None:
                    matches = bf.match(des_prev, des_curr)
                    matches = sorted(matches, key=lambda x: x.distance)
                    if matches:
                        idx = matches[0].trainIdx
                        p0 = np.array([[keypoints_curr[idx].pt]], dtype=np.float32)
                        self.latest_position = (int(p0[0][0][0]), int(p0[0][0][1]))

                        if not self.headless:
                            cv2.circle(frame, self.latest_position, 5, (0, 0, 255), -1)
                        # print("Point recovered.")
                    else:
                        # print("Recovery failed. Using last known position.")
                        p0 = np.array([[self.latest_position[0], self.latest_position[1]]], dtype=np.float32)
                else:
                    # print("Descriptors not found. Using last known position.")
                    p0 = np.array([[self.latest_position[0], self.latest_position[1]]], dtype=np.float32)

            if not self.headless:
                cv2.imshow('Point Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

            frame_gray = frame_gray_new.copy()

        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        """Stops the tracking process."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

    def get_latest_position(self):
        """Returns the latest tracked position."""
        return self.latest_position

    def get_latest_position_in_percentage_from_center(self):
        """
        Get the latest position of the tracked point in percentage coordinates.
        """
        return self.latest_position[0] / self.image_size[0], self.latest_position[1] / self.image_size[1]


if __name__ == "__main__":
    tracker = PointTracker((200, 300), headless=False)
    try:
        while tracker.running:
            time.sleep(1)
    finally:
        tracker.stop()
