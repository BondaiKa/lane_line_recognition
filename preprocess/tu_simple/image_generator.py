from tensorflow.keras.utils import Sequence
from dotenv import load_dotenv


class SimpleFrameGenerator(Sequence):
    pass


class GeneratorCreator:
    pass


if __name__ == "__main__":
    load_dotenv()
    AMOUNT_OF_FRAMES = 10000
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2

    tu_simple_image_generator = GeneratorCreator