import os
import sys
import torch
from tqdm import tqdm
from typing import Union, List, Tuple

from dataset_manager import DatasetManager
from model_handler import ModelHandler

class HiddenStateDataManager:
    def __init__(
        self,
        dataset_manager: DatasetManager,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        output_path: str,
        use_separate_system_message: bool,
        batch_size: int = 8,
        quantization_bits: int = 4
    ):
        self.model_handler = None
        self.dataset_hidden_states = []
        self.batch_size = batch_size

        filename = output_path + "_hidden_state_samples.pt"
        if os.path.exists(filename):
            print(f"Loading existing '{filename}'... ", end="")
            sys.stdout.flush()
            self.load_hidden_state_samples(filename)
            print(f"Done ({self.get_total_samples()} samples; {self.get_num_layers()} layers).")
        else:
            self._load_model(pretrained_model_name_or_path, quantization_bits=quantization_bits)
            dataset_tokens = self._tokenize_datasets(dataset_manager, use_separate_system_message)
            self._generate_hidden_state_samples(dataset_tokens)
            print(f"Saving to '{filename}'... ", end="")
            sys.stdout.flush()
            self.save_hidden_state_samples(filename)
            print("Done.")

    def _load_model(self, pretrained_model_name_or_path: Union[str, os.PathLike], quantization_bits: int = 4):
        try:
            self.model_handler = ModelHandler(
                pretrained_model_name_or_path,
                device="cuda",
                quantization_bits=quantization_bits
            )
        except Exception as e:
            print(f"Error loading model: {e}")

    def _tokenize_datasets(
        self,
        dataset_manager: DatasetManager,
        use_separate_system_message: bool
    ) -> List[List[dict]]:
        """
        dataset_tokens[i][j] = A dictionary returned by apply_chat_template, containing 'input_ids' and possibly other keys.
        'input_ids' should be of shape (1, seq_len) for each prompt.
        """
        dataset_tokens = [[] for _ in range(dataset_manager.get_num_classes())]
        try:
            with tqdm(total=dataset_manager.get_total_samples(), desc="Tokenizing prompts") as bar:
                for i, dataset in enumerate(dataset_manager.datasets):
                    for system_message, prompt in dataset:
                        if use_separate_system_message:
                            conversation = [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt}
                            ]
                        else:
                            conversation = [{"role": "user", "content": system_message + " " + prompt}]
                        tokens = self.model_handler.tokenizer.apply_chat_template(
                            conversation=conversation,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        dataset_tokens[i].append(tokens)
                        bar.update(n=1)
        except Exception as e:
            print(f"Error during tokenization: {e}")
        return dataset_tokens

    def _prepare_batch(
        self,
        token_list: List[dict],
        start_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Extract a batch of token dictionaries
        batch_data = token_list[start_idx:start_idx + self.batch_size]

        # Find max length from 'input_ids'
        max_len = max(d['input_ids'].size(1) for d in batch_data)

        batch_size = len(batch_data)
        device = self.model_handler.model.device

        # Initialize padded tensors
        # Left-pad sequences with the pad_token_id (or eos if pad not defined)
        pad_id = self.model_handler.tokenizer.pad_token_id if self.model_handler.tokenizer.pad_token_id is not None else self.model_handler.tokenizer.eos_token_id
        padded_tokens = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        # Fill in actual tokens at the right side of each sequence
        for i, tokens_dict in enumerate(batch_data):
            input_ids = tokens_dict['input_ids']  # shape: (1, seq_len)

            if input_ids.dim() == 1:
                # Add a batch dimension so it becomes (1, seq_len)
                input_ids = input_ids.unsqueeze(0)

            seq_len = input_ids.size(1)
            # We place them at the end
            padded_tokens[i, -seq_len:] = input_ids[0, :seq_len]
            attention_mask[i, -seq_len:] = 1

        return padded_tokens, attention_mask

    def _generate(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Generate a single token for each sequence in the batch and extract the hidden states.
        output.hidden_states[-1] is the final step (the newly generated token).
        Each element in output.hidden_states[-1] is (batch, seq_len_total, hidden_size).
        The newly generated token is at index -1 in seq_len_total.
        """
        output = self.model_handler.model.generate(
            tokens,
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
            pad_token_id=self.model_handler.tokenizer.pad_token_id if self.model_handler.tokenizer.pad_token_id is not None else self.model_handler.tokenizer.eos_token_id
        )

        # Test check if a new token was generated
        if output.sequences.size(1) == tokens.size(1):
            # Here just skip:
            print("WARNING! NO NEW TOKENS GENERATED")
            return [[] for _ in range(tokens.size(0))]

        batch_size = tokens.size(0)
        batch_hidden_states = []
        final_step = output.hidden_states[-1]  # tuple of layers
        # final_step[i]: (batch, seq_len, hidden_size)

        # Verify shapes...
        for layer_idx, hidden_state in enumerate(final_step):
            assert hidden_state.dim() == 3, f"Layer {layer_idx} hidden state is not 3D: {hidden_state.shape}"
            assert hidden_state.size(0) == batch_size, f"Batch size mismatch at layer {layer_idx}"
            assert hidden_state.size(1) > 0, f"No tokens present at layer {layer_idx}"

        for batch_idx in range(batch_size):
            # Extract layer hidden states for the last generated token
            hidden_states_by_layer = [
                layer_hs[batch_idx, -1, :].cpu()
                for layer_hs in final_step
            ]

            # Compute deltas
            deltas = [
                hidden_states_by_layer[i] - hidden_states_by_layer[i - 1]
                for i in range(1, len(hidden_states_by_layer))
            ]

            batch_hidden_states.append(deltas)

        return batch_hidden_states

    def _generate_hidden_state_samples(self, dataset_tokens: List[List[dict]]) -> None:
        """
        Process all datasets and collect their hidden states.
        We assume each class dataset is balanced and each sample leads to one new token generation.
        After processing, self.dataset_hidden_states will contain:
          self.dataset_hidden_states[class_idx][sample_idx][layer] = (hidden_size) tensor of layer deltas
        """
        try:
            total_samples = sum(len(tokens) for tokens in dataset_tokens)

            # Sanity check - ensure class sizes are equal
            class_sizes = [len(tokens) for tokens in dataset_tokens]
            if len(set(class_sizes)) != 1:
                print(f"WARNING: Uneven class sizes detected: {class_sizes}")

            with tqdm(total=total_samples, desc="Sampling hidden states") as bar:
                for class_idx, token_list in enumerate(dataset_tokens):
                    hidden_states = []
                    expected_samples = len(token_list)

                    for start_idx in range(0, len(token_list), self.batch_size):
                        batch_tokens, attention_mask = self._prepare_batch(token_list, start_idx)
                        batch_hidden_states = self._generate(batch_tokens, attention_mask)

                        # Check we got the expected batch size results
                        batch_actual_size = min(self.batch_size, len(token_list) - start_idx)
                        if len(batch_hidden_states) != batch_actual_size:
                            print(f"WARNING: Batch size mismatch in class {class_idx}, batch starting at {start_idx}")
                            print(f"Expected: {batch_actual_size}, Got: {len(batch_hidden_states)}")

                        hidden_states.extend(batch_hidden_states)
                        bar.update(n=batch_actual_size)

                    # Final check for this class
                    if len(hidden_states) != expected_samples:
                        print(f"WARNING: Final sample count mismatch for class {class_idx}")
                        print(f"Expected: {expected_samples}, Got: {len(hidden_states)}")

                    self.dataset_hidden_states.append(hidden_states)

            # Final validation after all processing
            if len(self.dataset_hidden_states) != len(dataset_tokens):
                print("WARNING: Missing classes in final dataset")
                print(f"Expected {len(dataset_tokens)} classes, got {len(self.dataset_hidden_states)}")

            # Check layer consistency
            if self.dataset_hidden_states:
                expected_layers = len(self.dataset_hidden_states[0][0])
                for class_idx, class_states in enumerate(self.dataset_hidden_states):
                    for sample_idx, sample_states in enumerate(class_states):
                        if len(sample_states) != expected_layers:
                            print(f"WARNING: Layer count mismatch in class {class_idx}, sample {sample_idx}")
                            print(f"Expected {expected_layers} layers, got {len(sample_states)}")

        except Exception as e:
            print(f"Error generating hidden states: {e}")

    def get_datasets(self, layer_index: int) -> List[torch.Tensor]:
        return [torch.stack([sample[layer_index] for sample in dataset]) for dataset in self.dataset_hidden_states]

    def get_differenced_datasets(self, layer_index: int) -> List[torch.Tensor]:
        datasets = self.get_datasets(layer_index)
        return [dataset - datasets[0] for dataset in datasets[1:]]

    def get_num_layers(self) -> int:
        return len(self.dataset_hidden_states[0][0])

    def get_num_dataset_types(self) -> int:
        return len(self.dataset_hidden_states)

    def get_total_samples(self) -> int:
        return sum(len(dataset) for dataset in self.dataset_hidden_states)

    def get_num_features(self, layer_index: int) -> int:
        return self.dataset_hidden_states[0][0][layer_index].shape[-1]

    def load_hidden_state_samples(self, file_path: str) -> None:
        try:
            self.dataset_hidden_states = torch.load(file_path)
        except Exception as e:
            print(f"Error loading hidden state samples from {file_path}: {e}")

    def save_hidden_state_samples(self, file_path: str) -> None:
        try:
            torch.save(self.dataset_hidden_states, file_path)
        except Exception as e:
            print(f"Error saving hidden state samples to {file_path}: {e}")

