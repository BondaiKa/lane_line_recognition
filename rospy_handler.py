import rospy
from .base import AbstractVideoHandler, FrameHandler
from geometry_msgs.msg import PoseArray
import cv2
from typing import Union

BASE_CAMERA_NUMBER = 2


class RospyVideoHandler(AbstractVideoHandler):
    """Using real-time video and do lane line recognition"""

    def __init__(self, camera_path: Union[str, int] = BASE_CAMERA_NUMBER,
                 init_node: str = "points_publisher",
                 publisher: str = "/campoints"):
        super().__init__(camera_path=camera_path)
        rospy.init_node(init_node, anonymous=True)
        self.points_pub = rospy.Publisher(publisher, PoseArray, queue_size=10)
        self.rate = rospy.Rate(10)

        # TODO @Karim: try to understand meaning variables
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def process(self):
        """Process image with filter and transform image"""
        while not rospy.is_shutdown():
            ret, frame_dis = self.video.read()

            frame_handler = FrameHandler()
            frame = frame_handler.preprocess_frame(frame=frame_dis)
            points, num_classes = frame_handler.recognize(frame=frame)
            result = frame_handler.draw_popylines(frame=frame, points=points, labels=num_classes)
            cv2.imshow('cap', result)

        cv2.destroyAllWindows()
