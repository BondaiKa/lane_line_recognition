from base import AbstractVideoHandler, FrameHandler
import cv2
import logging

log = logging.getLogger(__name__)


class FakeVideoHandler(AbstractVideoHandler):
    """Draw lane line recognition for fake videos"""

    def process(self):
        """Process image with filter and transform image"""
        log.info('Start fake video processing...')
        while self.video.isOpened():
            ret, frame_dis = self.video.read()

            # frame_handler = FrameHandler()
            # frame = frame_handler.preprocess_frame(frame=frame_dis)
            # points, num_classes = frame_handler.recognize(frame=frame)
            # result = frame_handler.draw_popylines(frame=frame, points=points, labels=num_classes)
            cv2.imshow('cap', frame_dis)
            cv2.waitKey(0)
        log.info("Destroy all windows...")
        cv2.destroyAllWindows()
