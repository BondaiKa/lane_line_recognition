from base import AbstractVideoHandler, FrameHandler
import cv2
import logging

log = logging.getLogger(__name__)


class FakeVideoHandler(AbstractVideoHandler):
    """Draw lane line recognition for fake videos"""

    def process(self):
        """Process image with filter and transform image"""
        log.info('Start fake video processing...')
        while self.video.isOpened(): #TODO @Karim revert to full video not 1 frame
            ret, frame_dis = self.video.read()
            cv2.imshow(f'initial_{frame_dis.shape}', frame_dis)
            # width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width, height = 1280, 960
            log.debug(f"Input resolution params: Width: {width}, height: {height}")
            log.debug(f"Frame shape: {frame_dis.shape}")
            frame_handler = FrameHandler()
            frame = frame_handler.preprocess_frame(frame=frame_dis, width=width, height=height)
            polylines, labels = frame_handler.recognize(frame=frame)
            result = frame_handler.draw_popylines(frame=frame, points=polylines, labels=labels)
            cv2.imshow(f'Final frame_{result}', result)
            cv2.waitKey(1)

        log.info("Destroy all windows...")
        cv2.destroyAllWindows()
