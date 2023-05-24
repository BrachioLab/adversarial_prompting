import torch
HUGGING_FACE_TOKEN = "TOKEN_HERE"

PREPEND_TASK_VERSIONS = {}
PREPEND_TASK_VERSIONS[1] = "a picture of a dog"
PREPEND_TASK_VERSIONS[2] = "a picture of a mountain"
PREPEND_TASK_VERSIONS[3] = "a picture of the ocean"

PATH_TO_LLAMA_WEIGHTS = "PATH_TO_LLAMA" # Weights are assumed to be in PATH/llama-7b, PATH/llama-13b, etc.
PATH_TO_LLAMA_TOKENIZER = "PATH_TO_LLAMA"
PATH_TO_VICUNA = "PATH_TO_VICUNA"

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def trimmed_mean(tensor, p=0.2, dim=1):

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    n = sorted_tensor.size(dim)
    trim_elements = int(n * p)

    if trim_elements > 0:
        trimmed_tensor = sorted_tensor.narrow(dim, trim_elements, n - 2 * trim_elements)
    else:
        # Display warning the first time this happens
        if not hasattr(trimmed_mean, "warning_displayed"):
            print("Warning: trimmed_mean is trimming 0 elements. This may be due to a small n_init_per_prompt or a small p.")
            trimmed_mean.warning_displayed = True

        trimmed_tensor = sorted_tensor

    return trimmed_tensor.mean(dim=dim)