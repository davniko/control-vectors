import os
import sys
import torch

from tqdm import tqdm

from typing import Union, List

from dataset_manager import DatasetManager
from model_handler import ModelHandler

class HiddenStateDataManager:

    def __init__(
        self,
        dataset_manager: DatasetManager,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        output_path: str,
        use_separate_system_message: bool,
        batch_size: int = 1,
        use_bfloat16: bool = True
    ):
        self.model_handler = None
        self.dataset_hidden_states = []
        self.batch_size = batch_size
        self.use_bfloat16 = use_bfloat16

        filename = output_path + "_hidden_state_samples.pt"
        if os.path.exists(filename):
            print(f"Loading existing '{filename}'... ", end="")
            sys.stdout.flush()
            self.load_hidden_state_samples(filename)
            print(f"Done ({self.get_total_samples()} samples; {self.get_num_layers()} layers).")
        else:
            self._load_model(pretrained_model_name_or_path)
            dataset_tokens = self._tokenize_datasets(dataset_manager, use_separate_system_message)
            self._generate_hidden_state_samples(dataset_tokens)
            print(f"Saving to '{filename}'... ", end="")
            sys.stdout.flush()
            self.save_hidden_state_samples(filename)
            print("Done.")
    
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

    def _load_model(self, pretrained_model_name_or_path: Union[str, os.PathLike]):
        try:
            self.model_handler = ModelHandler(pretrained_model_name_or_path, device = "cuda", use_bfloat16 = self.use_bfloat16)
        except Exception as e:
            print(f"Error loading model: {e}")

    def _tokenize_datasets(
        self,
        dataset_manager: DatasetManager,
        use_separate_system_message: bool
    ) -> List[List[torch.Tensor]]:
        dataset_tokens = [[] for _ in range(dataset_manager.get_num_classes())]
        try:
            with tqdm(total = dataset_manager.get_total_samples(), desc = "Tokenizing prompts") as bar:
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
                            conversation = conversation,
                            add_generation_prompt = True,
                            return_tensors = "pt"
                        )
                        dataset_tokens[i].append(tokens)
                        bar.update(n = 1)
        except Exception as e:
            print(f"Error during tokenization: {e}")
        return dataset_tokens

    def _generate_hidden_state_samples(self, dataset_tokens: List[List[torch.Tensor]]) -> None:
        try:
            num_samples = sum(len(tokens) for tokens in dataset_tokens)
            with tqdm(total = num_samples, desc = "Sampling hidden states") as bar:
                for token_list in dataset_tokens:
                    hidden_states = []
                    
                    if self.batch_size <= 1:
                        for tokens in token_list:
                            hidden_states.append(self._generate(tokens))
                            bar.update(n = 1)
                    else:
                        sorted_indices = sorted(range(len(token_list)), key=lambda i: token_list[i].size(1))
                        sorted_tokens = [token_list[i] for i in sorted_indices]
                        
                        for i in range(0, len(sorted_tokens), self.batch_size):
                            batch_tokens = sorted_tokens[i:i+self.batch_size]
                            
                            try:
                                batch_hidden_states = self._generate_batch(batch_tokens)
                                hidden_states.extend(batch_hidden_states)
                                bar.update(n = len(batch_tokens))
                            except Exception as e:
                                print(f"Error processing batch: {e}")
                                # Fallback
                                for tokens in batch_tokens:
                                    try:
                                        hidden_states.append(self._generate(tokens))
                                        bar.update(n = 1)
                                    except Exception as e:
                                        print(f"Error processing individual token: {e}")
                    
                    # attempt smaller batch size
                    if self.batch_size > 1:
                        unsort_mapping = {idx: original_idx for original_idx, idx in enumerate(sorted_indices)}
                        ordered_hidden_states = [None] * len(hidden_states)
                        for i, hs in enumerate(hidden_states):
                            ordered_hidden_states[unsort_mapping[i]] = hs
                        hidden_states = ordered_hidden_states
                    
                    self.dataset_hidden_states.append(hidden_states)
        except Exception as e:
            print(f"Error generating hidden states: {e}")

    def _generate(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        output = self.model_handler.model.generate(
            tokens.to(self.model_handler.model.device),
            use_cache = False,
            max_new_tokens = 1,
            return_dict_in_generate = True,
            output_hidden_states = True,
            attention_mask = torch.ones(tokens.size(), dtype=torch.long).to(tokens.device),
            pad_token_id = self.model_handler.tokenizer.pad_token_id if self.model_handler.tokenizer.pad_token_id is not None else self.model_handler.tokenizer.eos_token_id
        )
        hidden_states_by_layer = [hidden_state[:, -1,:].squeeze().to('cpu') for hidden_state in output.hidden_states[-1][:]]
        deltas = [hidden_states_by_layer[i] - hidden_states_by_layer[i - 1] for i in range(1, len(hidden_states_by_layer))]
        return deltas

    def _generate_batch(self, tokens_batch: List[torch.Tensor], max_retries: int = 3) -> List[List[torch.Tensor]]:
        # Retry logic is for handling OOM errors
        for retry in range(max_retries):
            try:
                max_length = max(tokens.size(1) for tokens in tokens_batch)
                padded_tokens = []
                attention_masks = []
                
                for tokens in tokens_batch:
                    seq_len = tokens.size(1)
                    padding_length = max_length - seq_len
                    pad_token_id = self.model_handler.tokenizer.pad_token_id if self.model_handler.tokenizer.pad_token_id is not None else self.model_handler.tokenizer.eos_token_id
                    padded = torch.full((1, max_length), pad_token_id, dtype=tokens.dtype, device=tokens.device)
                    padded[:, padding_length:] = tokens
                    padded_tokens.append(padded)                    
                    mask = torch.zeros_like(padded)
                    mask[:, padding_length:] = 1
                    attention_masks.append(mask)
                
                batch_tokens = torch.cat(padded_tokens, dim=0)
                batch_attention_mask = torch.cat(attention_masks, dim=0)
                
                output = self.model_handler.model.generate(
                    batch_tokens.to(self.model_handler.model.device),
                    use_cache = False,
                    max_new_tokens = 1,
                    return_dict_in_generate = True,
                    output_hidden_states = True,
                    attention_mask = batch_attention_mask.to(self.model_handler.model.device),
                    pad_token_id = pad_token_id
                )
                
                batch_deltas = []                
                for i in range(len(tokens_batch)):
                    hidden_states_by_layer = [hidden_state[i, -1, :].squeeze().to('cpu') for hidden_state in output.hidden_states[-1][:]]
                    deltas = [hidden_states_by_layer[j] - hidden_states_by_layer[j - 1] for j in range(1, len(hidden_states_by_layer))]
                    batch_deltas.append(deltas)
                
                return batch_deltas
                
            except RuntimeError as e:
                # Try to handle CUDA OOM
                if "CUDA out of memory" in str(e):
                    if retry < max_retries - 1:
                        torch.cuda.empty_cache()
                        print(f"CUDA out of memory (attempt {retry+1}/{max_retries}). Reducing batch size...")
                        
                        if len(tokens_batch) == 1:
                            print("Processing large example on CPU...")
                            original_device = self.model_handler.model.device
                            try:
                                self.model_handler.model = self.model_handler.model.to('cpu')
                                result = self._generate(tokens_batch[0])
                                return [result]
                            finally:
                                self.model_handler.model = self.model_handler.model.to(original_device)
                        
                        elif len(tokens_batch) > 1:
                            mid = len(tokens_batch) // 2
                            print(f"Splitting batch from {len(tokens_batch)} to {mid} and {len(tokens_batch)-mid}...")
                            first_half = self._generate_batch(tokens_batch[:mid], max_retries=max_retries-retry)
                            second_half = self._generate_batch(tokens_batch[mid:], max_retries=max_retries-retry)
                            return first_half + second_half
                    else:
                        print(f"Failed after {max_retries} attempts. Falling back to single sample processing.")
                else:
                    raise
                    
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error in batch processing (attempt {retry+1}/{max_retries}): {e}")
                    if len(tokens_batch) > 1:
                        mid = len(tokens_batch) // 2
                        first_half = self._generate_batch(tokens_batch[:mid], max_retries=max_retries-retry)
                        second_half = self._generate_batch(tokens_batch[mid:], max_retries=max_retries-retry)
                        return first_half + second_half
                else:
                    print(f"Batch processing failed. Falling back to single processing.")
        
        # all retries failed - process individually as last resort
        results = []
        for tokens in tokens_batch:
            try:
                results.append(self._generate(tokens))
            except Exception as e:
                print(f"Failed to process individual sample: {e}")
        
        return results