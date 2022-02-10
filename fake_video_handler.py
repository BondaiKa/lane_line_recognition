from base import AbstractVideoHandler, FrameHandler
import cv2
import logging

log = logging.getLogger(__name__)


class FakeVideoHandler(AbstractVideoHandler):
    """Draw lane line recognition for fake videos"""

    def process(self, frame_handler: FrameHandler):
        """Process image with filter and transform image"""
        log.info('Start fake video processing...')
        while self.video.isOpened():  # TODO @Karim revert to full video not 1 frame
            ret, frame_dis = self.video.read()
            cv2.imshow(f'initial_frame', frame_dis)
            # width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame = frame_handler.preprocess_frame(frame=frame_dis)
            polyline_1, polyline_2, label_1, label_2 = frame_handler.recognize(frame=frame)
            polylines, colors = frame_handler.postprocess_frame(polylines=[polyline_1, polyline_2],
                                                                labels=[label_1, label_2])
            result = frame_handler.draw_popylines(frame=frame, list_of_points=polylines, list_of_colors=colors)
            cv2.imshow(f'Final frame', result)
            cv2.waitKey(1)

        log.info("Destroy all windows...")
        cv2.destroyAllWindows()
