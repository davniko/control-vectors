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
            triplets_per_batch: int = 8,
            quantization_bits: int = 4
    ):
        """
        Args:
            dataset_manager: The DatasetManager that loads all classes (0=baseline, 1=neg, 2=pos, etc.)
            pretrained_model_name_or_path: HF model path
            output_path: prefix for saving hidden states
            use_separate_system_message: whether to treat the system prompt as separate
            triplets_per_batch: how many "baseline / negative / positive" triplets to process in each batch.
            quantization_bits: 4 or 8
        """
        self.model_handler = None
        self.dataset_hidden_states = []
        self.triplets_per_batch = triplets_per_batch

        filename = output_path + "_hidden_state_samples.pt"
        if os.path.exists(filename):
            print(f"Loading existing '{filename}'... ", end="")
            sys.stdout.flush()
            self.load_hidden_state_samples(filename)
            print(f"Done ({self.get_total_samples()} samples; {self.get_num_layers()} layers).")
        else:
            self._load_model(pretrained_model_name_or_path, quantization_bits)
            dataset_tokens = self._tokenize_datasets(dataset_manager, use_separate_system_message)
            self._generate_hidden_state_samples(dataset_tokens)
            print(f"Saving to '{filename}'... ", end="")
            sys.stdout.flush()
            self.save_hidden_state_samples(filename)
            print("Done.")

    def _load_model(self, model_path: Union[str, os.PathLike], quantization_bits: int):
        try:
            self.model_handler = ModelHandler(
                model_path,
                device="cuda",
                quantization_bits=quantization_bits
            )
            self.model_handler.tokenizer.padding_side = 'left'
        except Exception as e:
            print(f"Error loading model: {e}")

    def _tokenize_datasets(
            self,
            dataset_manager: DatasetManager,
            use_separate_system_message: bool
    ) -> List[List[dict]]:
        """
        Returns a list of length num_classes.
        dataset_tokens[class_id] is a list[dict], each dict containing 'input_ids', etc.
        The i-th dict in baseline class is *supposed* to align with the i-th dict in the negative class, etc.
        """
        dataset_tokens = [[] for _ in range(dataset_manager.get_num_classes())]

        total = dataset_manager.get_total_samples()
        with tqdm(total=total, desc="Tokenizing prompts") as bar:
            for class_idx, dataset in enumerate(dataset_manager.datasets):
                for system_message, prompt in dataset:
                    if use_separate_system_message:
                        conversation = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ]
                    else:
                        conversation = [
                            {"role": "user", "content": system_message + " " + prompt}
                        ]

                    tokens = self.model_handler.tokenizer.apply_chat_template(
                        conversation=conversation,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    dataset_tokens[class_idx].append(tokens)
                    bar.update(1)

        return dataset_tokens

    def _prepare_triplet_batch(
            self,
            dataset_tokens: List[List[dict]],
            start_idx: int
    ) -> Tuple[List[Tuple[int, dict]], int]:
        """
        Gathers up to self.triplets_per_batch triplets of (baseline, negative, positive).
        Returns:
          batch_list, actual_batch_count
        where batch_list is a list of (class_id, token_dict) for each sample
        in the order: baseline_0, negative_0, positive_0, baseline_1, negative_1, positive_1, ...
        The actual_batch_count is how many triplets we found (0..triplets_per_batch).

        This function assumes you have exactly 3 classes: [0=baseline, 1=negative, 2=positive].
        If you have more classes, adapt the logic accordingly.
        """
        if len(dataset_tokens) < 3:
            raise ValueError("Triplet-based batching requires at least 3 classes (baseline/neg/pos).")

        baseline_len = len(dataset_tokens[0])
        negative_len = len(dataset_tokens[1])
        positive_len = len(dataset_tokens[2])

        max_triplets_available = min(baseline_len, negative_len, positive_len) - start_idx
        if max_triplets_available <= 0:
            return [], 0

        num_triplets = min(self.triplets_per_batch, max_triplets_available)

        batch_list = []
        for i in range(num_triplets):
            idx = start_idx + i
            batch_list.append((0, dataset_tokens[0][idx]))
            batch_list.append((1, dataset_tokens[1][idx]))
            batch_list.append((2, dataset_tokens[2][idx]))

        return batch_list, num_triplets

    def _prepare_padded_tensors(
            self,
            batch_list: List[Tuple[int, dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Takes a list of (class_id, token_dict) items,
        merges them into a single batch of size len(batch_list),
        returns (padded_tokens, attention_mask, class_ids).

        We do genuine LEFT padding: (left side = pad tokens, real tokens on the right).
        """
        count = len(batch_list)
        if count == 0:
            return None, None, []

        max_len = 0
        for _, tk in batch_list:
            seq_len = tk['input_ids'].size(1)
            if seq_len > max_len:
                max_len = seq_len

        device = self.model_handler.model.device
        pad_id = (self.model_handler.tokenizer.pad_token_id
                  if self.model_handler.tokenizer.pad_token_id is not None
                  else self.model_handler.tokenizer.eos_token_id)

        padded_tokens = torch.full((count, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((count, max_len), dtype=torch.long, device=device)
        class_ids = []

        for i, (c_id, tk_dict) in enumerate(batch_list):
            class_ids.append(c_id)
            input_ids = tk_dict['input_ids']  # (1, seq_len)
            if input_ids.dim() == 2 and input_ids.size(0) == 1:
                input_ids = input_ids.squeeze(0)  # (seq_len,)
            seq_len = input_ids.size(0)

            padded_tokens[i, -seq_len:] = input_ids
            attention_mask[i, -seq_len:] = 1

        return padded_tokens, attention_mask, class_ids

    def _generate(
            self,
            tokens: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor or None:
        """
        For each row in tokens, generate 1 new token.
        Return shape (batch, num_layers - 1, hidden_size) with “layer deltas,” or None if no new token is generated.
        """
        output = self.model_handler.model.generate(
            tokens,
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
            pad_token_id=(self.model_handler.tokenizer.pad_token_id
                          if self.model_handler.tokenizer.pad_token_id is not None
                          else self.model_handler.tokenizer.eos_token_id)
        )

        if output.sequences.size(1) == tokens.size(1):
            # Means no new tokens across the entire batch
            print("WARNING: No new tokens generated for this batch => skipping.")
            return None

        final_step = output.hidden_states[-1]  # (num_layers, batch, seq_len, hidden_size)

        hidden_states_by_layer = [
            layer_hs[:, -1, :].cpu()  # (batch, hidden_size)
            for layer_hs in final_step
        ]
        stacked = torch.stack(hidden_states_by_layer, dim=0)  # (num_layers, batch, hidden_size)
        deltas = stacked[1:] - stacked[:-1]  # (num_layers-1, batch, hidden_size)
        deltas = deltas.permute(1, 0, 2).contiguous()  # (batch, num_layers-1, hidden_size)
        return deltas

    def _generate_hidden_state_samples(
            self,
            dataset_tokens: List[List[dict]]
    ) -> None:
        """
        We assume exactly 3 classes: baseline (0), negative (1), positive (2).
        We'll do triplets_per_batch at a time => each batch is 3 * triplets_per_batch items.
        Then we put the resulting deltas in self.dataset_hidden_states[ class_id ] in the correct order.
        """
        num_classes = len(dataset_tokens)
        self.dataset_hidden_states = [[] for _ in range(num_classes)]

        start_idx = 0
        total_triplets = min(len(dataset_tokens[0]), len(dataset_tokens[1]), len(dataset_tokens[2]))

        total_possible_samples = total_triplets * 3
        with tqdm(total=total_possible_samples, desc="Sampling hidden states") as bar:

            while start_idx < total_triplets:
                # Grab up to 'triplets_per_batch' triplets
                batch_list, actual_trip_count = self._prepare_triplet_batch(dataset_tokens, start_idx)
                if actual_trip_count == 0:
                    break  # no more to process

                # Convert that batch_list => padded tokens
                padded_tokens, attention_mask, class_ids = self._prepare_padded_tensors(batch_list)
                if padded_tokens is None:
                    break  # empty batch

                # Generate
                deltas_batch = self._generate(padded_tokens, attention_mask)
                batch_size = len(batch_list)  # should be 3 * actual_trip_count if all 3 classes were present

                if deltas_batch is not None:
                    # Shape => (batch_size, num_layers - 1, hidden_size)
                    # Distribute back to the correct class in the same order
                    for i in range(batch_size):
                        c_id = class_ids[i]
                        self.dataset_hidden_states[c_id].append(deltas_batch[i])
                    bar.update(batch_size)
                else:
                    # skip these items
                    bar.update(batch_size)

                start_idx += actual_trip_count

        # Final warnings if lengths differ
        lens_arrays = [len(x) for x in self.dataset_hidden_states]
        if len(set(lens_arrays)) != 1:
            print("WARNING: mismatch among classes in final sample counts:", lens_arrays)

    def get_datasets(self, layer_index: int) -> List[torch.Tensor]:
        """
        Return a list of Tensors (one per class).
        Each Tensors is shape [num_samples_in_class, hidden_size].
        Because each sample in self.dataset_hidden_states[class_id][sample_idx]
        is shape (#layers-1, hidden_size), we index [layer_index] to pick the row we want.
        """
        out = []
        for class_id, class_data in enumerate(self.dataset_hidden_states):
            # class_data = list of Tensors, each shape (#layers-1, hidden_size)
            # we want to gather the layer_index row from each sample => shape = [#samples, hidden_size]
            all_samples_for_class = torch.stack(
                [sample[layer_index] for sample in class_data],
                dim=0
            )
            out.append(all_samples_for_class)
        return out

    def get_differenced_datasets(self, layer_index: int) -> List[torch.Tensor]:
        """
        Subtract the baseline class (class 0) from each other class.
        Return a list of Tensors for classes [1..], each shape [num_samples, hidden_size].
        """
        all_data = self.get_datasets(layer_index)
        baseline = all_data[0]
        # classes 1..n
        return [d - baseline for d in all_data[1:]]

    def get_num_layers(self) -> int:
        """
        Each sample for each class is shape (#layers-1, hidden_size).
        So the "number of layers" is (#layers-1).
        We replicate the old code's approach (some spinoffs add +1).
        """
        if not self.dataset_hidden_states or not self.dataset_hidden_states[0]:
            return 0
        # pick the first sample of class 0
        return self.dataset_hidden_states[0][0].size(0)

    def get_num_dataset_types(self) -> int:
        """Number of classes."""
        return len(self.dataset_hidden_states)

    def get_total_samples(self) -> int:
        """Sum of sample counts across all classes."""
        return sum(len(x) for x in self.dataset_hidden_states)

    def get_num_features(self, layer_index: int) -> int:
        """
        E.g. shape (#layers-1, hidden_size) => hidden_size is the last dimension.
        """
        if not self.dataset_hidden_states or not self.dataset_hidden_states[0]:
            return 0
        return self.dataset_hidden_states[0][0][layer_index].size(0)

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


