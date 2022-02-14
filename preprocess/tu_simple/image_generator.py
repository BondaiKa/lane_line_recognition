from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv
from lane_line_recognition.base import AbstractFrameGenerator, AbstractFrameGeneratorCreator
from typing import Type
import os
from os.path import join, dirname


class TuSimpleFrameGenerator(AbstractFrameGenerator, Sequence):
    pass


class TuSimpleFrameGeneratorCreator(AbstractFrameGeneratorCreator):
    def get_generator(self) -> Type[TuSimpleFrameGenerator]:
        return TuSimpleFrameGenerator


if __name__ == "__main__":
    ENV_FILE_NAME = 'tu_simple.env'
    dotenv_path = join(dirname(__file__), ENV_FILE_NAME)
    load_dotenv(dotenv_path)


    AMOUNT_OF_FRAMES = 10000
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    TU_SIMPLE_FRAME_DATASET_PATH = os.getenv('TU_SIMPLE_FRAME_DATASET_PATH')
    TU_SIMPLE_JSON_HDF5_DATASET_PATH = os.getenv('TU_SIMPLE_JSON_HDF5_DATASET_PATH')
    MAX_LINES_PER_FRAME = int(os.getenv('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(os.getenv('MAX_NUM_POINTS'))

    tu_simple_image_generator = TuSimpleFrameGeneratorCreator(
        validation_split=0.2,
        frame_glob_path=TU_SIMPLE_FRAME_DATASET_PATH,
        json_hdf5_glob_path=TU_SIMPLE_JSON_HDF5_DATASET_PATH,
    )

    tu_simple_train_generator = tu_simple_image_generator.flow_from_directory(
        subset='training', shuffle=True, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES, max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS, num_type_of_lines=NUM_TYPE_OF_LINES,
        target_shape=input_shape,
    )

    tu_simple_validation_generator = tu_simple_image_generator.flow_from_directory(
        subset='validation', shuffle=True, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES, max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS, num_type_of_lines=NUM_TYPE_OF_LINES,
        target_shape=input_shape,
    )
