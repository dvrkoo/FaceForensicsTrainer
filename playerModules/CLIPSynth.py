"""code from cozzolino et all"""

import torch
import os
import numpy as np
import yaml
from PIL import Image

from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from playerModules.CLIP_net.utils import make_normalize
from playerModules.CLIP_net import create_architecture, load_weights

from scipy.special import logsumexp, logit, log_expit


softplusinv = lambda x: np.log(np.expm1(x))  # log(exp(x)-1)
softminusinv = lambda x: x - np.log(
    -np.expm1(x)
)  # info: https://jiafulow.github.io/blog/2019/07/11/softplus-and-softminus/

fusion_functions = {
    "mean_logit": lambda x, axis: np.mean(x, axis),
    "max_logit": lambda x, axis: np.max(x, axis),
    "median_logit": lambda x, axis: np.median(x, axis),
    "lse_logit": lambda x, axis: logsumexp(x, axis),
    "mean_prob": lambda x, axis: softminusinv(
        logsumexp(log_expit(x), axis) - np.log(x.shape[axis])
    ),
    "soft_or_prob": lambda x, axis: -softminusinv(np.sum(log_expit(-x), axis)),
}


def apply_fusion(x, typ, axis):
    return fusion_functions[typ](x, axis)


def get_config(model_name, weights_dir="./weights"):
    with open(os.path.join(weights_dir, model_name, "config.yaml")) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data["weights_file"])
    return (
        data["model_name"],
        model_path,
        data["arch"],
        data["norm_type"],
        data["patch_size"],
    )


def run_single_image(image_path, weights_dir, models_list, device):
    """
    Run inference on a single image for the given list of models.

    Parameters:
      image_path (str): Path to the input image.
      weights_dir (str): Directory where model weights are stored.
      models_list (list): List of model names to load.
      device (torch.device): Device to run inference on.

    Returns:
      dict: A dictionary mapping model names to their prediction logit.
    """

    # Dictionaries to hold models and their corresponding transforms
    models_dict = {}
    transform_dict = {}

    print("Loading models:")
    for model_name in models_list:
        print(f"Loading {model_name}...", flush=True)
        # get_config is assumed to return (other, model_path, arch, norm_type, patch_size)
        _, model_path, arch, norm_type, patch_size = get_config(
            model_name, weights_dir=weights_dir
        )

        # Create and load the model
        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

        # Build transformation steps based on patch_size and norm_type
        transform_steps = []
        if patch_size is None:
            print("Using no resizing/cropping.", flush=True)
            transform_key = f"none_{norm_type}"
        elif patch_size == "Clip224":
            print("Resizing to Clip224...", flush=True)
            transform_steps.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform_steps.append(CenterCrop((224, 224)))
            transform_key = f"Clip224_{norm_type}"
        elif isinstance(patch_size, (tuple, list)):
            print(f"Resizing to {patch_size}...", flush=True)
            transform_steps.append(Resize(*patch_size))
            transform_steps.append(CenterCrop(patch_size[0]))
            transform_key = f"res{patch_size[0]}_{norm_type}"
        elif patch_size > 0:
            print(f"Center cropping with size {patch_size}...", flush=True)
            transform_steps.append(CenterCrop(patch_size))
            transform_key = f"crop{patch_size}_{norm_type}"
        else:
            raise ValueError("Unsupported patch_size value")

        # Append normalization step
        transform_steps.append(make_normalize(norm_type))
        transform = Compose(transform_steps)

        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)

    # Load the single image and convert it to RGB
    image = Image.open(image_path).convert("RGB")

    # Apply each transform (one per unique key) to the image, add a batch dimension
    transformed_imgs = {}
    for key, transform in transform_dict.items():
        tensor_img = transform(image)
        # Add batch dimension (resulting shape: (1, C, H, W))
        transformed_imgs[key] = tensor_img.unsqueeze(0)

    # Run inference on each model and store the results
    results = {}
    with torch.no_grad():
        for model_name, (transform_key, model) in models_dict.items():
            input_tensor = transformed_imgs[transform_key].clone().to(device)
            out_tens = model(input_tensor).cpu().numpy()

            # Process the output as in the original function
            if out_tens.shape[1] == 1:
                out_tens = out_tens[:, 0]
            elif out_tens.shape[1] == 2:
                out_tens = out_tens[:, 1] - out_tens[:, 0]
            else:
                raise ValueError("Unexpected output shape from model")

            # If the output has spatial dimensions, average them
            if len(out_tens.shape) > 1:
                logit = np.mean(out_tens, axis=(1, 2))
                # With a single image, extract the single value
                logit = logit[0]
            else:
                logit = out_tens[0]

            results[model_name] = logit

    return results


def predict(image_path):
    args = {}
    args["device"] = torch.device("cuda" if torch.cuda.is_available() else "mps")
    args["fusion"] = "soft_or_prob"
    args["models"] = ["clipdet_latent10k_plus", "Corvi2023"]
    # print current dir
    args["weights_dir"] = "./playerModules/CLIP_WEIGHTS/"

    results = run_single_image(
        image_path, args["weights_dir"], args["models"], args["device"]
    )
    # rename model names to be more clear
    results["CLIPDet"] = results.pop("clipdet_latent10k_plus")
    results["Corvi"] = results.pop("Corvi2023")
    logits_array = np.array(list(results.values()))

    # Now, you can apply fusion. For a 1D array, axis=0 or axis=-1 is the same.
    results["Final Score"] = apply_fusion(logits_array, "soft_or_prob", axis=0)

    # transform the results from np.float32 to float
    results = {k: float(v) for k, v in results.items()}
    return results
