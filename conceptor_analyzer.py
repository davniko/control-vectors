# code for computing conceptors borrowed from https://github.com/jorispos/ConceptorSteering/

import torch
import logging
from typing import Union, Optional, List
from hidden_state_data_manager import HiddenStateDataManager
from tqdm import tqdm


def scale_conceptor(C: torch.Tensor, aperture_scaling_factor: float) -> torch.Tensor:
    """
    Scales a conceptor matrix using the given aperture scaling factor.

    Args:
        C: Conceptor matrix of shape (d, d)
        aperture_scaling_factor: Positive scaling factor

    Returns:
        Scaled conceptor matrix of shape (d, d)
    """
    return C @ torch.inverse(C + (aperture_scaling_factor**-2) * (torch.eye(C.shape[0], device=C.device) - C))


def rescale_conceptor(C, prev_alpha, new_alpha):
    return scale_conceptor(C, new_alpha / prev_alpha)


def compute_conceptor(X, aperture):
    """
    Computes the conceptor matrix for a given input matrix X.
    (PyTorch version)

    Parameters:
    - X (torch.Tensor): Input matrix of shape (n_samples, n_features).
    - torch.Tensor: Conceptor matrix of shape (n_features, n_features).
    """
    R = torch.matmul(X.T, X) / X.shape[0]
    U, S, _ = torch.svd(R)
    C = U * (S / (S + (aperture ** (-2)) * torch.ones(S.shape, device=X.device))) @ U.T
    return C


def combine_conceptors_and(C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
    """
    Combines two conceptors C1 and C2 using the given formula. (AND operation, does not work so well)

    Parameters:
    - C1 (torch.Tensor): First conceptor tensor of shape (n_features, n_features).
    - C2 (torch.Tensor): Second conceptor tensor of shape (n_features, n_features).

    Returns:
    - torch.Tensor: Combined conceptor tensor of shape (n_features, n_features).
    """
    I = torch.eye(C1.shape[0], device=C1.device)  # Identity matrix
    C1_inv = torch.inverse(C1)
    C2_inv = torch.inverse(C2)
    combined_inv = C1_inv + C2_inv - I
    combined = torch.inverse(combined_inv)
    return combined


def combine_conceptors(C1, C2):
    """
    TODO: what is this?
    Combines two conceptors C1 and C2 using the given new formula. (OR operation which works much better than AND)

    Parameters:
    - C1 (torch.Tensor): First conceptor tensor of shape (n_features, n_features).
    - C2 (torch.Tensor): Second conceptor tensor of shape (n_features, n_features).

    Returns:
    - torch.Tensor: Combined conceptor tensor of shape (n_features, n_features).
    """
    I = torch.eye(C1.shape[0], device=C1.device)  # Identity matrix
    I_C1_inv = torch.inverse(I - C1)
    I_C2_inv = torch.inverse(I - C2)
    combined_inv = I_C1_inv + I_C2_inv - I
    combined = torch.inverse(combined_inv)
    result = I - combined
    return result


def compute_conceptor_low_rank(X: torch.Tensor, aperture: float, rank: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank approximation of the conceptor matrix for given input matrix X.

    Parameters:
    - X (torch.Tensor): Input matrix of shape (n_samples, n_features).
    - aperture (float): Aperture parameter.
    - rank (int): Rank for the approximation.

    Returns:
    - U (torch.Tensor): Left singular vectors of shape (d, rank).
    - S_c (torch.Tensor): Scaled singular values of shape (rank,).
    """
    # Compute covariance matrix
    R = torch.matmul(X.T, X) / X.shape[0]

    # Perform truncated SVD
    U, S, V = torch.svd_lowrank(R, q=rank)

    # Scale singular values as per conceptor formula
    S_c = S / (S + aperture ** (-2))

    return U, S_c


def compute_low_rank_conceptor_automatic(X: torch.Tensor, aperture: float, variance_threshold: float=0.99) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank approximation of the conceptor matrix using a variance threshold
    to select the rank.

    Parameters:
        X (torch.Tensor): Input matrix of shape (n_samples, n_features).
        aperture (float): Aperture parameter.
        variance_threshold (float): Desired variance retention.

    Returns:
        U_k (torch.Tensor): Left singular vectors (n_features, k).
        s_k (torch.Tensor): Singular values (k,).
    """
    # Compute the correlation matrix
    R = torch.matmul(X.T, X) / X.shape[0]

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(R)

    # Reverse to get descending order
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    # Compute scaling factors
    scaling_factors = eigenvalues / (eigenvalues + aperture ** (-2))

    # Cumulative variance
    cumulative_variance = torch.cumsum(scaling_factors, dim=0)
    total_variance = cumulative_variance[-1]
    variance_ratios = cumulative_variance / total_variance

    # Determine k
    k = torch.searchsorted(variance_ratios, variance_threshold).item() + 1

    U_k = eigenvectors[:, :k]
    s_k = scaling_factors[:k]

    return U_k, s_k


def compute_optimal_low_rank_conceptor(C: torch.Tensor, threshold: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank approximation of a conceptor matrix by keeping only
    significant singular values.

    Args:
        C: Original conceptor matrix
        threshold: Relative threshold for keeping singular values
                  (compared to largest singular value)

    Returns:
        U: Left singular vectors (d × k)
        s: Singular values (k,)
        where k is the determined rank
    """
    U, s, _ = torch.svd(C)

    # Find optimal rank by looking at singular value decay
    max_sv = s[0]
    mask = s >= (threshold * max_sv)
    k = mask.sum().item()

    # Keep only the top k components
    U_k = U[:, :k]
    s_k = s[:k]

    return U_k, s_k


def reconstruct_conceptor(U: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Reconstructs full conceptor matrix from low-rank representation."""
    return U @ torch.diag(s) @ U.T


class ConceptorAnalyzer:
    """
    Computes conceptors for each dataset (class) and layer, optionally after mean-centering.

    Produces conceptors[class_idx][layer_idx] => (d x d) or None,
    plus a parallel structure means[class_idx][layer_idx] => (d,) or None
    if we want to store the mean used.

    The 'center_mode' can be:
      - 'none': no mean-centering
      - 'local': subtract each dataset's own mean
      - 'baseline': subtract the baseline dataset's mean from all other classes
         (assuming class_idx=0 is 'baseline').
    """

    def __init__(
            self,
            hidden_state_data_manager: "HiddenStateDataManager",
            skip_begin_layers: Union[int, float] = 0,
            skip_end_layers: Union[int, float] = 1,
            aperture: float = 0.1,
            center_mode: str = "none",
            low_rank_approximation: bool = False,
            low_rank_method: Optional[str] = None,
            rank: Optional[int] = None,
            variance_threshold: Optional[float] = None,
            threshold: Optional[float] = None,
    ):
        """
        Args:
            hidden_state_data_manager: The data manager that supplies get_datasets(layer).
            skip_begin_layers (int|float): # of initial layers to skip (may be fraction).
            skip_end_layers (int|float): # of final layers to skip (may be fraction).
            aperture (float): Aperture parameter for the conceptor formula (must be > 0).
            center_mode (str): "none", "local", or "baseline"
                               controlling how we perform mean-centering.
        """
        if aperture <= 0:
            raise ValueError("Aperture must be positive")
        self.hidden_state_data_manager = hidden_state_data_manager

        self.num_layers = hidden_state_data_manager.get_num_layers()
        self.num_dataset_types = hidden_state_data_manager.get_num_dataset_types()

        if isinstance(skip_begin_layers, float) and 0 < skip_begin_layers < 1:
            skip_begin_layers = round(skip_begin_layers * self.num_layers)
        if isinstance(skip_end_layers, float) and 0 < skip_end_layers < 1:
            skip_end_layers = round(skip_end_layers * self.num_layers)

        if skip_begin_layers + skip_end_layers >= self.num_layers:
            raise ValueError("Too many layers skipped (start + end >= total layers).")

        self.skip_begin_layers = skip_begin_layers
        self.skip_end_layers = skip_end_layers

        self.aperture = aperture
        self.center_mode = center_mode.lower()
        if self.center_mode not in ["none", "local", "baseline"]:
            raise ValueError(f"Invalid center_mode: {self.center_mode}")

        self.low_rank_approximation = low_rank_approximation

        if self.low_rank_approximation:
            if low_rank_method not in ["manual", "automatic", "optimal"]:
                raise ValueError("low_rank_method must be one of 'manual', 'automatic', or 'optimal' when low_rank_approximation is True")
            self.low_rank_method = low_rank_method
            if low_rank_method == "manual":
                if rank is None:
                    raise ValueError("When low_rank_method is 'manual', 'rank' must be specified")
                self.rank = rank
            elif low_rank_method == "automatic":
                if variance_threshold is None:
                    raise ValueError("When low_rank_method is 'automatic', 'variance_threshold' must be specified")
                self.variance_threshold = variance_threshold
            elif low_rank_method == "optimal":
                if threshold is None:
                    raise ValueError("When low_rank_method is 'optimal', 'threshold' must be specified")
                self.threshold = threshold
            else:
                raise ValueError(f"Unknown low_rank_method {low_rank_method}")
        else:
            self.low_rank_method = None
            self.rank = None
            self.variance_threshold = None
            self.threshold = None

        self.conceptors: List[List[Optional[torch.Tensor]]] = []
        self.means: List[List[Optional[torch.Tensor]]] = []

        self._compute_conceptors_all()

    def _compute_conceptors_all(self):
        """
        Iterates over dataset classes and layers to compute conceptors.

        If center_mode='local', we subtract each dataset's own mean.
        If center_mode='baseline', we subtract the baseline's mean for classes>0.
        If center_mode='none', no mean is subtracted.
        """

        # Initialize output structures
        self.conceptors = [[None for _ in range(self.num_layers)]
                           for _ in range(self.num_dataset_types)]
        self.means = [[None for _ in range(self.num_layers)]
                      for _ in range(self.num_dataset_types)]

        # Prepass for 'baseline' mode...
        baseline_means_by_layer = [None] * self.num_layers
        if self.center_mode == "baseline" and self.num_dataset_types > 0:
            print("Computing baseline means...")
            for layer_idx in range(self.skip_begin_layers, self.num_layers - self.skip_end_layers):
                X_base = self.hidden_state_data_manager.get_datasets(layer_idx)[0]  # baseline is class_idx=0
                X_base = X_base.float().cpu()
                baseline_means_by_layer[layer_idx] = X_base.mean(dim=0, keepdim=True)  # shape (1, d)

        total_computations = (self.num_layers - self.skip_begin_layers - self.skip_end_layers) * self.num_dataset_types

        computation_counter = 0
        with tqdm(total=total_computations, desc="Computing conceptors") as pbar:
            for class_idx in range(self.num_dataset_types):
                print(f"\nProcessing class {class_idx + 1}/{self.num_dataset_types}:")
                for layer_idx in range(self.skip_begin_layers, self.num_layers - self.skip_end_layers):
                    print(f"- Layer {layer_idx + 1}/{self.num_layers}: ", end="")

                    X = self.hidden_state_data_manager.get_datasets(layer_idx)[class_idx]
                    X = X.float().cpu()
                    N_samples = X.shape[0]
                    print(f"{N_samples} samples", end="")

                    # Mean-Centering
                    if self.center_mode == "none":
                        mean_vec = None
                    elif self.center_mode == "local":
                        mean_vec = X.mean(dim=0, keepdim=True)
                        X = X - mean_vec
                    elif self.center_mode == "baseline":
                        # Don't shift if this is the baseline class...
                        if class_idx == 0:
                            mean_vec = None
                        else:
                            mean_vec = baseline_means_by_layer[layer_idx]
                            X = X - mean_vec
                    else:
                        raise ValueError(f"Unknown center_mode {self.center_mode}")

                    try:
                        if self.low_rank_approximation:
                            if self.low_rank_method == "manual":
                                U_k, s_k = compute_conceptor_low_rank(X, self.aperture, rank=self.rank)
                            elif self.low_rank_method == "automatic":
                                U_k, s_k = compute_low_rank_conceptor_automatic(X, self.aperture, variance_threshold=self.variance_threshold)
                            elif self.low_rank_method == "optimal":
                                C_full = compute_conceptor(X, self.aperture)
                                U_k, s_k = compute_optimal_low_rank_conceptor(C_full, threshold=self.threshold)
                            else:
                                raise ValueError(f"Unknown low_rank_method {self.low_rank_method}")

                            # Store low-rank components
                            self.conceptors[class_idx][layer_idx] = (U_k.cpu(), s_k.cpu())

                            rank_k = U_k.shape[1]
                            print(f"[rank: {rank_k}/{X.shape[1]}] ", end="")
                            print(f"[var: {(s_k.sum() / X.shape[1] * 100):.1f}%] ", end="")
                            print(f"[λ: ({s_k.min():.3f}, {s_k.max():.3f})]", end="")
                        else:
                            C = compute_conceptor(X, self.aperture)
                            self.conceptors[class_idx][layer_idx] = C.cpu()
                            print(" [Full Matrix]", end="")
                            print(f"[trace: {torch.trace(C).item():.2f}] ", end="")
                            # print(f"[λ: ({eigenvals.min():.3f}, {eigenvals.max():.3f})]", end="")

                        print("")

                    except RuntimeError as e:
                        logging.error(f"Error computing conceptor at layer={layer_idx} class={class_idx}: {e}")
                        print(" [Computation Error]")

                    if mean_vec is not None:
                        self.means[class_idx][layer_idx] = mean_vec.squeeze(0).cpu()

                    computation_counter += 1
                    pbar.update(1)

    def get_conceptor(self, class_idx: int, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Returns the conceptor for the given class/layer, or None if not computed.
        """
        return self.conceptors[class_idx][layer_idx]

    def get_mean(self, class_idx: int, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Returns the (d,)-shaped mean vector used for that (class,layer), or None if none was used.
        """
        return self.means[class_idx][layer_idx]
