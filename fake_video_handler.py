from lane_line_recognition.base import AbstractVideoHandler, FrameHandler
import cv2
import logging
from base import transform_frame


log = logging.getLogger(__name__)


class FakeVideoHandler(AbstractVideoHandler):
    """Draw lane line recognition for fake videos"""

    def process(self, frame_handler: FrameHandler):
        """Process image with filter and transform image"""
        log.info('Start fake video processing...')
        while self.video.isOpened():  # TODO @Karim revert to full video not 1 frame
            ret, frame_dis = self.video.read()
            # cv2.imshow(f'initial_frame', frame_dis)
            # final_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            # final_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame = frame_handler.preprocess_frame(frame=frame_dis)
            polylines, labels = frame_handler.recognize(frame=frame)
            labels_probability = frame_handler.get_labels_probability(labels)
            full_polylines, full_colors = frame_handler.postprocess_frame(polylines=polylines, labels=labels,
                                                                          width=frame_dis.shape[1],
                                                                          height=frame_dis.shape[0])
            #TODO @Karim: rescale polyline after reducing values in the dataset
            full_result = frame_handler.draw_probability(frame=frame_dis, labels_probability=labels_probability)
            full_result = frame_handler.draw_popylines(frame=full_result, list_of_points=full_polylines,
                                                       list_of_colors=full_colors)
            cv2.imshow(f'Final frame', full_result)

            presp_frame = transform_frame(full_result, frame_dis.shape[1], frame_dis.shape[0])
            cv2.imshow('Perspective transform frame', presp_frame)

            cv2.waitKey(1)

        log.info("Destroy all windows...")
        cv2.destroyAllWindows()
