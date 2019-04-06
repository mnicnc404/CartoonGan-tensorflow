import os
from tensorflow.keras import models
from keras_contrib.layers import InstanceNormalization


PRETRAINED_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pretrained_models")


def load_model(style):
    custom_objects = {'InstanceNormalization': InstanceNormalization}

    style = style.replace("comic", "")
    return models.load_model(
        os.path.join(PRETRAINED_MODEL_DIR, f"generator_model{style}.h5"),
        custom_objects,
        compile=False
    )


if __name__ == "__main__":
    import PIL
    import numpy as np

    custom_objects = {'InstanceNormalization': InstanceNormalization}
    models.load_model(
        os.path.join(PRETRAINED_MODEL_DIR, f"generator_model{_type}.h5"), custom_objects)

    image_path = "tmp/test.jpg"

    input_image = PIL.Image.open(image_path).convert("RGB")
    input_image = np.asarray(input_image)
    input_image = input_image.astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)

    input_image = (input_image / 255 * 2) - 1

    transformed_image = model(input_image)
    # print(transformed_image)
    print(f"min: {np.amin(transformed_image)}")
    print(f"max: {np.amax(transformed_image)}")

    print("-" * 50)
    print(transformed_image.shape)

    transformed_image = transformed_image.numpy()
    transformed_image = transformed_image[0]


    transformed_image = ((transformed_image + 1) / 2) * 255

    print(transformed_image)


    image = PIL.Image.fromarray(transformed_image.astype("uint8"))
    image.save("tmp/test_out.jpg")