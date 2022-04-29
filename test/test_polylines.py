from lane_line_recognition.base import draw_polyline_in_frame
from lane_line_recognition.base.polyline_image_generator import PolylineLaneFrameGeneratorCreator
from lane_line_recognition.base.utils import test_generator
from dotenv import load_dotenv, find_dotenv, dotenv_values
import os

if __name__ == "__main__":
    from dotenv import load_dotenv

    ENV_FOLDER = 'env'
    BASE_ENV = '.env'
    TEST_POLYLINE_ENV_FILE_NAME = 'test_polyline.env'

    root_folder = os.path.split(os.path.dirname(__file__))[0]
    polyline_dotenv_path = os.path.join(root_folder, ENV_FOLDER, TEST_POLYLINE_ENV_FILE_NAME)
    base_dotenv = os.path.join(root_folder, BASE_ENV)

    config = {
        **dotenv_values(base_dotenv),
        **dotenv_values(polyline_dotenv_path),
    }

    AMOUNT_OF_FRAMES = 10000
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    MAX_LINES_PER_FRAME = int(config.get('MAX_LINES_PER_FRAME'))
    MAX_NUM_POINTS = int(config.get('MAX_NUM_POINTS'))
    NUM_TYPE_OF_LINES = int(config.get('NUM_TYPE_OF_LINES'))
    FINAL_WIDTH = int(config.get('FINAL_WIDTH'))
    FINAL_HEIGHT = int(config.get('FINAL_HEIGHT'))
    IMAGE_PATH = config.get('FRAME_DATASET_PATH')
    JSON_HDF5_DATASET_PATH = config.get('JSON_HDF5_DATASET_PATH')

    input_shape = (FINAL_WIDTH, FINAL_HEIGHT, 1)
    image_glob_path = IMAGE_PATH + '*/*/*20.jpg'
    json_hdf5_glob_path = JSON_HDF5_DATASET_PATH + '*/*/*20.jpg.hdf5'

    data_gen = PolylineLaneFrameGeneratorCreator(
        validation_split=VALIDATION_SPLIT,
        frame_glob_path=[image_glob_path],
        json_hdf5_glob_path=[json_hdf5_glob_path],
    )

    train_generator = data_gen.flow_from_directory(
        subset='training', shuffle=False, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES, max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS, num_type_of_lines=NUM_TYPE_OF_LINES,
        final_shape=input_shape,
    )

    validation_generator = data_gen.flow_from_directory(
        subset='validation', shuffle=False, batch_size=BATCH_SIZE,
        number_files=AMOUNT_OF_FRAMES, max_lines_per_frame=MAX_LINES_PER_FRAME,
        max_num_points=MAX_NUM_POINTS, num_type_of_lines=NUM_TYPE_OF_LINES,
        final_shape=input_shape,
    )

    test_generator(
        train_generator,
        draw_line=True,
        input_shape=input_shape,
        max_lines_per_frame=MAX_LINES_PER_FRAME
    )
