import argparse
import gc
import sys
import signal
import torch

from model_handler import ModelHandler
from dataset_manager import DatasetManager
from hidden_state_data_manager import HiddenStateDataManager
from direction_analyzer import DirectionAnalyzer

def signal_handler(sig, frame):  # @UnusedVariable
    sys.exit(1)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def main(
    model_id,
    output_path,
    prompt_stems_file_path,
    continuations_file_path,
    writing_prompts_file_path,
    num_prompt_samples,
    use_separate_system_message,
    skip_begin_layers,
    skip_end_layers,
    discriminant_ratio_tolerance,
    batch_size,
    quantization_bits  
):
    signal.signal(signal.SIGINT, signal_handler)

    torch.inference_mode()
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)

    # Updated DatasetManager instantiation
    dataset_manager = DatasetManager(
        prompt_stems_file_path,
        continuations_file_path,
        writing_prompts_file_path,
        num_prompt_samples
    )

    hidden_state_data_manager = HiddenStateDataManager(
        dataset_manager,
        model_id,
        output_path,
        use_separate_system_message,
        batch_size,
        quantization_bits
    )

    direction_analyzer = DirectionAnalyzer(
        hidden_state_data_manager,
        skip_begin_layers,
        skip_end_layers,
        discriminant_ratio_tolerance
    )

    for i, direction_matrices_by_class in enumerate(direction_analyzer.direction_matrices):

        if any(direction_matrix_by_layer is not None for direction_matrix_by_layer in direction_matrices_by_class):

            # Free as much memory as possible and reload unquantized into system RAM.
            free_memory()
            model_handler = ModelHandler(
                model_id,
                device = "cpu",
                quantization_bits=quantization_bits
            )
            
            if i == 0:
                name = "debias"
            else:
                name = dataset_manager.class_names[i]
            
            # Save as control vectors in '.gguf' format.
            model_handler.export_gguf(direction_matrices_by_class, output_path + f"_{name}.gguf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify and save a model based on baseline, desired and undesired instructions.")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to load the pretrained model from.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the modified models to.")
    parser.add_argument("--prompt_stems_file", type=str, required=True, help="The file path for prompt stems.")
    parser.add_argument("--continuations_file", type=str, required=True, help="The file path for continuations.")
    parser.add_argument("--writing_prompts_file", type=str, required=True, help="The file path for writing prompts.")
    parser.add_argument("--num_prompt_samples", type = int, default = 10000, help = "The number of prompts to sample per class.")
    parser.add_argument("--use_separate_system_message", action="store_true", default=False, help="Use separate system message in conversation.")
    parser.add_argument("--skip_begin_layers", type = int, default = 0, help = "The number (or fraction) of initial layers to skip.")
    parser.add_argument("--skip_end_layers", type = int, default = 1, help = "The number (or fraction) of end layers to skip.")
    parser.add_argument("--discriminant_ratio_tolerance", type = float, default = 0.5, help = "Used to filter low signal \"noise\" directions (0 = none).")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to process in each batch")
    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Number of bits for quantization (4 or 8). Default is 4-bit as in original implementation."
    )
    args = parser.parse_args()
    
    main(
        args.model_id,
        args.output_path,
        args.prompt_stems_file,
        args.continuations_file,
        args.writing_prompts_file,
        args.num_prompt_samples,
        args.use_separate_system_message,
        args.skip_begin_layers,
        args.skip_end_layers,
        args.discriminant_ratio_tolerance,
        args.batch_size,
        args.quantization_bits
    )