import argparse
import gc
import sys
import signal
import torch

from model_handler import ModelHandler
from dataset_manager import DatasetManager
from hidden_state_data_manager import HiddenStateDataManager
from direction_analyzer import DirectionAnalyzer
from conceptor_analyzer import ConceptorAnalyzer

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
    use_conceptor,
    conceptor_aperture,
    center_mode,
    low_rank_approximation,
    low_rank_method,
    rank,
    variance_threshold,
    threshold
):
    signal.signal(signal.SIGINT, signal_handler)

    torch.inference_mode()
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)

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
        use_separate_system_message
    )

    if use_conceptor:
        print(f"==> Running ConceptorAnalyzer with aperture={conceptor_aperture} center_mode={center_mode}")
        c_analyzer = ConceptorAnalyzer(
            hidden_state_data_manager=hidden_state_data_manager,
            skip_begin_layers=skip_begin_layers,
            skip_end_layers=skip_end_layers,
            aperture=conceptor_aperture,
            center_mode=center_mode,
            low_rank_approximation=low_rank_approximation,
            low_rank_method=low_rank_method,
            rank=rank,
            variance_threshold=variance_threshold,
            threshold=threshold
        )
        conceptor_filename = output_path + "_conceptors.gguf"
        print(f"Saving computed conceptors and means to '{conceptor_filename}'...")

        model_handler = ModelHandler(model_id, device="cpu")
        model_handler.export_gguf_conceptors(c_analyzer.conceptors, c_analyzer.means, conceptor_filename)
        model_handler.delete()

    else:
        direction_analyzer = DirectionAnalyzer(
            hidden_state_data_manager,
            skip_begin_layers,
            skip_end_layers,
            discriminant_ratio_tolerance
        )

        for i, direction_matrices_by_class in enumerate(direction_analyzer.direction_matrices):
            if any(direction_matrix_by_layer is not None for direction_matrix_by_layer in direction_matrices_by_class):
                free_memory()
                model_handler = ModelHandler(model_id, device="cpu")
                if i == 0:
                    name = "debias"
                else:
                    name = dataset_manager.class_names[i]
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
    parser.add_argument("--use_conceptor", action="store_true", default=False,
                        help="Use conceptors instead of cross-cov vectors.")
    parser.add_argument("--conceptor_aperture", type=float, default=0.1,
                        help="Aperture for conceptor if using --use_conceptor.")
    parser.add_argument("--center_mode", type=str, default="none",
                        choices=["none", "local", "baseline"],
                        help="How to perform mean-centering during conceptor extraction.")
    parser.add_argument("--low_rank_approximation", action="store_true", default=False,
                        help="Use low-rank approximation for conceptors.")
    parser.add_argument("--low_rank_method", type=str, default=None,
                        choices=["manual", "automatic", "optimal"],
                        help="Method for low-rank approximation ('manual', 'automatic', 'optimal').")
    parser.add_argument("--rank", type=int, default=None,
                        help="Rank for 'manual' low-rank method.")
    parser.add_argument("--variance_threshold", type=float, default=None,
                        help="Variance threshold for 'automatic' low-rank method (e.g., 0.99).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for 'optimal' low-rank method (e.g., 1e-4).")

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
        args.use_conceptor,
        args.conceptor_aperture,
        args.center_mode,
        args.low_rank_approximation,
        args.low_rank_method,
        args.rank,
        args.variance_threshold,
        args.threshold
    )
