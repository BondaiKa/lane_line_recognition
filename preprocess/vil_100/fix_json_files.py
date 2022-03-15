import json
import os
import glob
import cv2
from typing import Dict, Union
import logging

log = logging.getLogger(__name__)

from lane_line_recognition.preprocess.vil_100.utils import Vil100Json


class JsonReviewer:
    def __init__(self, json_glob_path: str, frame_dataset_path: str):
        self.json_glob_path = sorted(glob.glob(json_glob_path))
        self.frame_dataset_path = frame_dataset_path

    @staticmethod
    def fix_json_file(json_file: Dict[str, Union[str, int]], frame_real_height: int, frame_real_width: int,
                      frame_path: str) -> Dict[str, Union[str, int]]:
        if json_file[Vil100Json.INFO][Vil100Json.HEIGHT] != frame_real_height or \
                json_file[Vil100Json.INFO][Vil100Json.WIDTH] != frame_real_width:
            log.debug(f'Different resolution! File name: `{frame_path}`.')
            json_file[Vil100Json.INFO][Vil100Json.HEIGHT] = frame_real_height
            json_file[Vil100Json.INFO][Vil100Json.WIDTH] = frame_real_width

        return json_file

    def exec(self):
        for json_file_path in self.json_glob_path:
            with open(json_file_path, 'r+') as f:
                json_file = json.load(f)
                image_path = json_file[Vil100Json.INFO][Vil100Json.IMAGE_PATH]
                full_path = self.frame_dataset_path + '/' + image_path
                frame = cv2.imread(full_path)
                height, width = frame.shape[0], frame.shape[1]

                json_file = self.fix_json_file(json_file, height, width, image_path)

                f.seek(0)
                json.dump(json_file, f)
                f.truncate()


if __name__ == '__main__':
    from dotenv import load_dotenv
    log = logging.getLogger(__name__)
    log.debug('Start working...')
    load_dotenv()
    JSON_PATH = os.getenv('JSON_DATASET_PATH')
    FRAME_DATASET_PATH = os.getenv('FRAME_DATASET_PATH')

    json_glob_path = JSON_PATH + '/*/*.json'

    reviewer = JsonReviewer(
        json_glob_path=json_glob_path,
        frame_dataset_path=FRAME_DATASET_PATH
    )
    reviewer.exec()
    log.debug('End working...')
