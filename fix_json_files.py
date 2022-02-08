import json
from dotenv import load_dotenv
import logging
import os
import glob
import cv2

from vil_100_utils import Vil100Json


class JsonReviewer:
    def __init__(self, json_glob_path: str, frame_dataset_path: str):
        self.json_glob_path = sorted(glob.glob(json_glob_path))
        self.frame_dataset_path = frame_dataset_path

    def exec(self):
        for json_file_path in self.json_glob_path:
            # TODO: check open format
            with open(json_file_path, 'r+') as f:
                json_file = json.load(f)
                image_path = json_file[Vil100Json.INFO][Vil100Json.IMAGE_PATH]
                full_path = self.frame_dataset_path + '/' + image_path
                frame = cv2.imread(full_path)
                log.debug(frame.shape)
                height, width = frame.shape[0],frame.shape[1]

                if json_file[Vil100Json.INFO][Vil100Json.HEIGHT] != height or \
                        json_file[Vil100Json.INFO][Vil100Json.WIDTH] != width:
                    log.warning(f'Different resolution! File name:{image_path}')
                    print(f'Different resolution! File name:{image_path}')

                    json_file[Vil100Json.INFO][Vil100Json.HEIGHT] = height
                    json_file[Vil100Json.INFO][Vil100Json.WIDTH] = width

                    f.seek(0)
                    json.dump(json_file, f)
                    f.truncate()


if __name__ == '__main__':
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
