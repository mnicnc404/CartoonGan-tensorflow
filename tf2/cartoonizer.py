import os
import PIL
import sys
import glob
import imageio
import logging
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from cartoongan import build_model

STYLES = ["shinkai", "hayao", "hosoda", "paprika"]
VALID_EXTENSIONS = ['jpg', 'png', 'gif']

# TODO: add self-trained cartoongan: MODE
# TODO: add documentation of each function
# TODO: readme, install guide(pip install -r requirements.txt + keras-contri?)
# TODO: adjust options' order
# TODO: fix screwy gif

parser = argparse.ArgumentParser(description="transform real world images to specified cartoon style(s)")
parser.add_argument("-s", "--styles", nargs="+", default=[STYLES[0]],
                    help="specify (multiple) cartoon styles which will be used to transform input images.")
parser.add_argument("-a", "--all_styles", action="store_true",
                    help="set true if all styled results are desired")
parser.add_argument("-i", "--input_dir", type=str, default="input_images",
                    help="directory with images to be transformed")
parser.add_argument("-o", "--output_dir", type=str, default="output_images",
                    help="directory where transformed images are saved")
parser.add_argument("-b", "--batch_size", type=int, default=1,
                    help="number of images that will be transformed in parallel to speed up processing. "
                         "higher value like 4 is recommended if there are gpus.")
parser.add_argument("--ignore_gif", action="store_true",
                    help="transforming gif images can take long time. enable this when you want to ignore gifs")
parser.add_argument("--overwrite", action="store_true",
                    help="enable this if you want to regenerate outputs regardless of existing results")
parser.add_argument("--skip_comparison", action="store_true",
                    help="enable this if you only want individual style result and to save processing time")
parser.add_argument("-v", "--comparison_view", type=str, default="horizontal",
                    choices=["horizontal", "vertical", "grid"],
                    help="specify how input images and transformed images are concatenated for easier comparison")
parser.add_argument("-f", "--gif_frame_frequency", type=int, default=2,
                    help="how often should a frame in gif be cartoonized. freq=1 means that every frame "
                         "in the gif will be cartoonized by default. set higher frequency can save processing "
                         "time while make the cartoonized gif less smooth")
parser.add_argument("-n", "--max_num_frames", type=int, default=100,
                    help="max number of frames that will be extracted from a gif. set higher value if longer gif "
                         "is needed")
parser.add_argument("--logging_lvl", type=str, default="info",
                    choices=["debug", "info", "warning", "error", "critical"],
                    help="logging level which decide how verbosely the program will be. set to `debug` if necessary")
parser.add_argument("--debug", action="store_true",
                    help="show the most detailed logging messages for debugging purpose")
parser.add_argument("--show_tf_cpp_log", action="store_true")

# TODO: limit image size
# TODO: processing mp4 possible? how about converting to mp4?

args = parser.parse_args()

TEMPORARY_DIR = f"{args.output_dir}/.tmp"


def pre_processing(image_path, expand_dim=True):
    input_image = PIL.Image.open(image_path).convert("RGB")
    input_image = np.asarray(input_image)
    input_image = input_image.astype(np.float32)
    input_image = input_image[:, :, [2, 1, 0]]
    if expand_dim:
        input_image = np.expand_dims(input_image, axis=0)
    # logger.debug(f"input_image.shape after pre-processing: {input_image.shape}")
    return input_image


def post_processing(transformed_image):
    if not type(transformed_image) == np.ndarray:
        transformed_image = transformed_image.numpy()
    transformed_image = transformed_image[0]
    transformed_image = transformed_image[:, :, [2, 1, 0]]
    transformed_image = transformed_image * 0.5 + 0.5
    transformed_image = transformed_image * 255
    return transformed_image


def save_transformed_image(output_image, img_filename, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    transformed_image_path = os.path.join(save_dir, img_filename)

    if output_image is not None:
        image = PIL.Image.fromarray(output_image.astype("uint8"))
        image.save(transformed_image_path)

    return transformed_image_path


def save_concatenated_image(image_paths, image_folder="comparison"):
    # TODO: add style as title
    images = [PIL.Image.open(i).convert('RGB') for i in image_paths]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    array = [np.asarray(i.resize(min_shape)) for i in images]

    if args.comparison_view == "horizontal":
        images_comb = np.hstack(array)
    elif args.comparison_view == "grid":
        if len(args.styles) + 1 == 4:
            first_row = np.hstack(array[:2])
            second_row = np.hstack(array[2:])
            images_comb = np.vstack([first_row, second_row])
        else:
            images_comb = np.hstack(array)
    else:
        images_comb = np.vstack(array)

    images_comb = PIL.Image.fromarray(images_comb)
    file_name = image_paths[0].split("/")[-1]

    if args.output_dir not in image_folder:
        image_folder = os.path.join(args.output_dir, image_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_path = os.path.join(image_folder, file_name)
    images_comb.save(image_path)
    return image_path


def convert_gif_to_png(gif_path):
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

    num_processed_frames = 0
    logger.debug("Generating png images...")
    try:
        while num_processed_frames < args.max_num_frames:

            image.putpalette(palette)
            extracted_image = PIL.Image.new("RGBA", image.size)
            extracted_image.paste(image)

            if i % args.gif_frame_frequency == 0:
                png_filename = f"{i + 1}.png"
                png_path = os.path.join(png_dir, png_filename)
                extracted_image.save(png_path)
                png_paths.append(png_path)
                num_processed_frames += 1

            image.seek(image.tell() + 1)
            i += 1

    except EOFError:
        pass  # end of sequence

    logger.debug(f"Number of {len(png_paths)} png images were generated at {png_dir}.")
    return png_paths


def transform_png_images(image_paths, model, style, return_existing_result=False):
    transformed_image_paths = list()
    save_dir = os.path.join("/".join(image_paths[0].split("/")[:-1]), style)
    logger.debug(f"Transforming {len(image_paths)} images and saving them to {save_dir}....")

    if return_existing_result:
        return glob.glob(os.path.join(save_dir, "*.png"))

    num_batch = int(np.ceil(len(image_paths) / args.batch_size))
    image_paths = np.array_split(image_paths, num_batch)

    logger.debug(f"Processing {num_batch} batches with batch_size={args.batch_size}...")
    for batch_image_paths in image_paths:
        image_filenames = [path.split("/")[-1] for path in batch_image_paths]
        input_images = [pre_processing(path, expand_dim=False) for path in batch_image_paths]
        input_images = np.stack(input_images, axis=0)
        transformed_images = model(input_images)
        output_images = [post_processing(image)
                         for image in np.split(transformed_images, transformed_images.shape[0])]
        paths = [save_transformed_image(img, f, save_dir)
                 for img, f in zip(output_images, image_filenames)]
        transformed_image_paths.extend(paths)

    return transformed_image_paths


def save_png_images_as_gif(image_paths, image_filename, style="comparison"):

    gif_dir = os.path.join(args.output_dir, style)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_path = os.path.join(gif_dir, image_filename)

    with imageio.get_writer(gif_path, mode='I') as writer:
        file_names = sorted(image_paths, key=lambda x: int(x.split('/')[-1].replace('.png', '')))
        #   file_names = file_names[::5]  # TODO: add option to pick every n frames
        logger.debug(f"Combining {len(file_names)} png images into {gif_path}...")
        last = -1
        for i, filename in enumerate(file_names):
            frame = 2 * (i ** 0.5)

            image = imageio.imread(filename)
            writer.append_data(image)


def result_exist(image_path, style):
    return os.path.exists(os.path.join(args.output_dir, style, image_path.split("/")[-1]))


def main():
    start = datetime.now()
    logger.info(f"Transformed images will be saved to `{args.output_dir}` folder.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create temporary folder which will be deleted after transformations
    if not os.path.exists(TEMPORARY_DIR):
        os.makedirs(TEMPORARY_DIR)

    # decide what styles to used in this execution
    styles = STYLES if args.all_styles else args.styles
    # TODO: check style input
    models = [build_model(s) for s in styles]

    logger.info(f"Cartoonizing images using {', '.join(styles)} style...")

    image_paths = []
    for ext in VALID_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))
    logger.info(f"Preparing to transform {len(image_paths)} images from `{args.input_dir}` directory...")

    progress_bar = tqdm(image_paths)
    for image_path in progress_bar:
        image_filename = image_path.split("/")[-1]
        progress_bar.set_description(f"Transforming {image_filename}")

        if image_filename.endswith(".gif") and not args.ignore_gif:
            png_paths = convert_gif_to_png(image_path)

            png_paths_list = [png_paths]
            num_images = len(png_paths)
            for model, style in zip(models, styles):
                return_existing_result = result_exist(image_path, style) or args.overwrite

                transformed_png_paths = transform_png_images(png_paths, model, style,
                                                             return_existing_result=return_existing_result)
                png_paths_list.append(transformed_png_paths)

                if not return_existing_result:
                    save_png_images_as_gif(transformed_png_paths, image_filename, style)

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
            input_image = pre_processing(image_path)

            for model, style in zip(models, styles):
                save_dir = os.path.join(args.output_dir, style)
                return_existing_result = result_exist(image_path, style) and not args.overwrite

                if not return_existing_result:
                    transformed_image = model(input_image)
                    output_image = post_processing(transformed_image)
                    transformed_image_path = save_transformed_image(output_image, image_filename, save_dir)
                else:
                    transformed_image_path = save_transformed_image(None, image_filename, save_dir)

                related_image_paths.append(transformed_image_path)

            if not args.skip_comparison:
                save_concatenated_image(related_image_paths)

    # TODO: decide wether to delete tmp dir

    # TODO: summary
    time_elapsed = datetime.now() - start
    logger.info(f"Total processing time: {time_elapsed}")


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

    if not args.show_tf_cpp_log:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    main()









