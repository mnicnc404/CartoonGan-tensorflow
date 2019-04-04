import argparse
import logging
import os
import PIL
import sys
import numpy as np
import glob
import imageio
from model import build_model

STYLES = ["shinkai", "hayao", "hosoda", "paprika"]
VALID_EXTENSIONS = ['jpg', 'png', 'gif']


# TODO: add documentation of each function
# TODO: readme, install guide
# TODO: 寫一個專門的 colab notebook 利用 GPU 作轉換


parser = argparse.ArgumentParser(description="cartoonize real world images to specified cartoon style")
parser.add_argument("-s", "--style", type=str, default="shinkai", choices=STYLES,
                    help="cartoon style to be used")
parser.add_argument("-a", "--all_style", action="store_true",
                    help="set true if all style result is desired")
parser.add_argument("-i", "--input_dir", type=str, default="input_images",
                    help="directory with images to be transformed")
parser.add_argument("-o", "--output_dir", type=str, default="output_images",
                    help="directory where transformed images are saved")
parser.add_argument("--ignore_gif", action="store_true",
                    help="enable this when you want to skip transforming gif image to save processing time")
parser.add_argument("--logging_lvl", type=str, default="info",
                    choices=["debug", "info", "warning", "error", "critical"])
parser.add_argument("--debug", action="store_true",
                    help="show the most detailed logging messages for debug purpose")
parser.add_argument("--overwrite", action="store_true",
                    help="enable this if you want to regenerate output regardless of existing results")
parser.add_argument("--skip_comparison", action="store_true",
                    help="enable this if you only want individual style result and to save processing time")

# TODO: gpu / cpu
# TODO: limit image size
# TODO: limit gif length

args = parser.parse_args()

TEMPORARY_DIR = f"{args.output_dir}/.tmp"


def pre_processing(image_path):
    input_image = PIL.Image.open(image_path).convert("RGB")
    input_image = np.asarray(input_image)
    input_image = input_image.astype(np.float32)
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = np.expand_dims(input_image, axis=0)
    # logger.debug(f"input_image.shape after pre-processing: {input_image.shape}")
    return input_image


def post_processing(transformed_image):
    transformed_image = transformed_image.numpy()
    transformed_image = transformed_image[0]
    transformed_image = transformed_image[:, :, [2, 1, 0]]
    transformed_image = transformed_image * 0.5 + 0.5
    transformed_image = transformed_image * 255
    return transformed_image


def save_transformed_image(output_image, img_filename, save_dir):
    image = PIL.Image.fromarray(output_image.astype("uint8"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    transformed_image_path = os.path.join(save_dir, img_filename)
    image.save(transformed_image_path)
    return transformed_image_path


def save_concatenated_image(image_paths, image_folder="comparison"):
    # TODO: add style as title
    images = [PIL.Image.open(i).convert('RGB') for i in image_paths]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    array = [np.asarray(i.resize(min_shape)) for i in images]
    images_comb = np.hstack(array)
    # TODO: add a parser option for horizontal / vertical concatenate

    # save that beautiful picture
    images_comb = PIL.Image.fromarray(images_comb)
    file_name = image_paths[0].split("/")[-1]

    if args.output_dir not in image_folder:
        image_folder = os.path.join(args.output_dir, image_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_path = os.path.join(image_folder, file_name)
    images_comb.save(image_path)
    return image_path


def convert_gif_to_png(gif_path, max_num_frames=100):
    logger.debug(f"`{gif_path}` is a gif, extracting png images from it...")
    gif_filename = gif_path.split("/")[-1].replace(".gif", "")
    image = PIL.Image.open(gif_path)
    palette = image.getpalette()
    png_paths = list()
    i = 0

    png_dir = os.path.join(TEMPORARY_DIR, gif_filename)
    if not os.path.exists(png_dir):
        logger.debug(f"Creating temporary folder: {png_dir} for storing intermediate result...")
        os.makedirs(png_dir)

    prev_generated_png_paths = glob.glob(png_dir + '/*.png')
    if prev_generated_png_paths:
        return prev_generated_png_paths

    logger.debug("Generating png images...")
    try:
        while i < max_num_frames:
            image.putpalette(palette)
            extracted_image = PIL.Image.new("RGBA", image.size)
            extracted_image.paste(image)

            png_filename = f"{i + 1}.png"
            png_path = os.path.join(png_dir, png_filename)
            extracted_image.save(png_path)
            png_paths.append(png_path)
            i += 1
            image.seek(image.tell() + 1)

    except EOFError:
        pass  # end of sequence

    logger.debug(f"Number of {len(png_paths)} png images were generated at {png_dir}.")
    return png_paths


def transform_png_images(image_paths, model, style):
    transformed_image_paths = list()
    save_dir = os.path.join("/".join(image_paths[0].split("/")[:-1]), style)
    logger.debug(f"Transforming {len(image_paths)} images and saving them to {save_dir}....")

    for image_path in image_paths:
        image_filename = image_path.split("/")[-1]

        input_image = pre_processing(image_path)
        transformed_image = model(input_image)
        output_image = post_processing(transformed_image)

        path = save_transformed_image(output_image, image_filename, save_dir)
        transformed_image_paths.append(path)

    return transformed_image_paths


def save_png_images_as_gif(image_paths, image_filename, style="comparison"):

    gif_dir = os.path.join(args.output_dir, style)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_path = os.path.join(gif_dir, image_filename)

    with imageio.get_writer(gif_path, mode='I') as writer:
        file_names = sorted(image_paths, key=lambda x: int(x.split('/')[-1].replace('.png', '')))
        #   file_names = file_names[::5]
        logger.debug(f"Combining {len(file_names)} png images into a single gif...")
        last = -1
        for i, filename in enumerate(file_names):
            frame = 2 * (i ** 0.5)

            image = imageio.imread(filename)
            writer.append_data(image)


def result_exist(image_path, style):
    return os.path.exists(os.path.join(args.output_dir, style, image_path.split("/")[-1]))


def main():

    logger.info(f"Transformed images will be saved to `{args.output_dir}` folder.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create temporary folder which will be deleted after transformations
    if not os.path.exists(TEMPORARY_DIR):
        os.makedirs(TEMPORARY_DIR)

    if args.all_style:
        styles = STYLES
        models = [build_model(s) for s in styles]
    else:
        style = args.style
        models = [build_model(style)]
        styles = [style]
    logger.info(f"Cartoonizing images using {', '.join(styles)} style...")

    image_paths = []
    for ext in VALID_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))
    logger.info(f"Preparing to transform {len(image_paths)} images from `{args.input_dir}` directory...")

    for image_path in image_paths:  # TODO: tqdm
        logger.info(f"Transforming {image_path}...")
        image_filename = image_path.split("/")[-1]

        if image_filename.endswith(".gif") and not args.ignore_gif:
            png_paths = convert_gif_to_png(image_path)

            png_paths_list = [png_paths]
            num_images = len(png_paths)
            for model, style in zip(models, styles):
                if result_exist(image_path, style) and not args.overwrite:
                    logger.debug("Skipping because result already exist and `overwrite` is disabled...")
                    continue

                transformed_png_paths = transform_png_images(png_paths, model, style)
                save_png_images_as_gif(transformed_png_paths, image_filename, style)
                png_paths_list.append(transformed_png_paths)

            rearrange_paths_list = [[l[i] for l in png_paths_list] for i in range(num_images)]

            save_dir = os.path.join(TEMPORARY_DIR, image_filename.replace(".gif", ""), "comparison")

            combined_image_paths = list()
            for image_paths in rearrange_paths_list:
                path = save_concatenated_image(image_paths, image_folder=save_dir)
                combined_image_paths.append(path)

            if not args.skip_comparison:
                save_png_images_as_gif(combined_image_paths, image_filename)

        else:
            related_image_paths = [image_path]
            # TODO: skip already-transformed images

            input_image = pre_processing(image_path)

            for model, style in zip(models, styles):
                if result_exist(image_path, style) and not args.overwrite:
                    logger.debug("Skipping because result already exist and `overwrite` is disabled...")
                    continue

                transformed_image = model(input_image)
                output_image = post_processing(transformed_image)

                save_dir = os.path.join(args.output_dir, style)
                transformed_image_path = save_transformed_image(output_image, image_filename, save_dir)
                related_image_paths.append(transformed_image_path)

            if not args.skip_comparison:
                save_concatenated_image(related_image_paths)

    # TODO: decide wether to delete tmp dir

    # TODO: summary


if __name__ == "__main__":

    logger = logging.getLogger("cartoonizer")
    logger.propagate = False
    log_lvl = {"debug": logging.DEBUG, "info": logging.INFO,
               "warning": logging.WARNING, "error": logging.ERROR,
               "critical": logging.CRITICAL}
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_lvl[args.logging_lvl])
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler(sys.stdout)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)

    main()







