import torch
import gc
import random
import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from .sae import create_sae_steering_vector, create_sae_steering_vector_latents

def clear_cuda(model=None):
    if model is not None:
        del model
    
    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def set_seed(seed=42):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def equalize_prompt_lengths(model, positive_prompt, neutral_prompt):
    positive_tokens = template_and_tokenize(model, positive_prompt).reshape(-1)
    neutral_tokens = template_and_tokenize(model, neutral_prompt).reshape(-1)

    original_positive_length = len(positive_tokens)
    original_neutral_length = len(neutral_tokens)

    if len(positive_tokens) > len(neutral_tokens):
        neutral_prompt += ' '
        while len(positive_tokens) > len(neutral_tokens):
            neutral_prompt += 'x'
            neutral_tokens = template_and_tokenize(model, neutral_prompt).reshape(-1)
    elif len(neutral_tokens) > len(positive_tokens):
        positive_prompt += ' '
        while len(neutral_tokens) > len(positive_tokens):
            positive_prompt += 'x'
            positive_tokens = template_and_tokenize(model, positive_prompt).reshape(-1)
    
    if original_positive_length != original_neutral_length:
        print('Trued up prompts to the same token length.')
        print('Neutral prompt is now:', neutral_prompt, 'with token length', len(neutral_tokens))
        print('Positive prompt is now:', positive_prompt, 'with token length', len(positive_tokens))
    
    return positive_tokens.reshape(1, -1), neutral_tokens.reshape(1, -1)


def template_and_tokenize(model, prompt):
    return model.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Hello!"},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
    )

def create_mean_caa_steering_vector(model, pos_tokens, neg_tokens, layer):
    """Extract and process activation differences using CAA. consider all token positions"""
    
    # Get activations for positive and negative sequences
    _, _, pos_cache = model.forward(pos_tokens, cache_activations_at=[layer])
    _, _, neg_cache = model.forward(neg_tokens, cache_activations_at=[layer])
    
    pos_acts = pos_cache[layer][0]
    neg_acts = neg_cache[layer][0]

    if len(pos_acts) > len(neg_acts):
        print('Warning! Pos acts longer than neg acts. Truncating. This works, but the length mismatch should have been handled in equalize_prompt_lengths.')
        diff = len(pos_acts) - len(neg_acts)
        pos_acts = pos_acts[:-diff]
    elif len(neg_acts) > len(pos_acts):
        print('Warning! Neg acts longer than pos acts. Truncating. This works, but the length mismatch should have been handled in equalize_prompt_lengths.')
        diff = len(neg_acts) - len(pos_acts)
        neg_acts = neg_acts[:-diff]

    # Find center point between positive and negative activations
    center = (pos_acts + neg_acts) / 2
    
    # Get vectors from center to positive point
    pos_vector = pos_acts - center
    
    # Average across all token positions (dim=0)
    pos_vector = pos_vector.mean(dim=0, keepdim=True)

    pos_vector = pos_vector / torch.norm(pos_vector)

    return pos_vector

def pca_directions(original_directions, n_components=None, whiten=False, svd_solver='auto', random_state=42):
    """Use PCA to extract steering direction from a set of directions
    
    Args:
        original_directions (torch.Tensor): The directions to process, shape [n_directions, d_model]
        n_components (int, float, str, None): Number of components to keep.
            If int, number of components to keep.
            If float between 0 and 1, fraction of variance to preserve.
            If 'mle', automatically determine components using MLE.
            If None, keep all components.
        whiten (bool): When True, the components_ vectors are multiplied by the 
            square root of n_samples and divided by the singular values to ensure 
            uncorrelated outputs with unit component-wise variances.
        svd_solver (str): SVD solver to use:
            'auto': automatically choose between 'full' and 'randomized'
            'full': run exact full SVD 
            'arpack': use arnoldi iteration
            'randomized': use randomized SVD
        random_state (int): Random seed for reproducibility
        
    Returns:
        torch.Tensor: The steering direction
    """

    # Convert to numpy for sklearn
    directions = original_directions.detach().float().cpu().numpy()
    
    # Reshape if needed - PCA expects [n_samples, n_features]
    if len(directions.shape) == 2 and directions.shape[0] == 1:
        directions = directions.reshape(1, -1)
    
    # Initialize PCA
    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver,
        random_state=random_state
    )

    # Fit and transform
    directions_pca = pca.fit_transform(directions)

    print('PCA explained variance ratio:', pca.explained_variance_ratio_)

    directions = pca.inverse_transform(directions_pca)

    # Convert back to torch tensor on original device
    directions = torch.from_numpy(directions).to(original_directions.device, original_directions.dtype)

    # Reshape back to original shape if needed
    # if len(processed_caa.shape) == 2 and processed_caa.shape[0] == 1:
    #     processed_caa = processed_caa.reshape(1, -1)
        
    # Normalize the processed vector
    if torch.norm(directions):
        directions = directions / torch.norm(directions)

    return directions

def pca_directions_with_mean_comparison(original_directions, n_components=None, whiten=False, svd_solver='auto', random_state=42):
    """TODO: tidy up
    Process vectors using both PCA and mean-scaling approaches for comparison"""
    
    # Convert to numpy for calculations
    vectors_np = original_directions.detach().float().cpu().numpy()
    
    # Calculate mean vector
    mean_vector = np.mean(vectors_np, axis=0)
    
    # Create mean-scaled reconstruction
    mean_scaled = np.zeros_like(vectors_np)
    scaling_factors = []
    
    # For each vector, find optimal scaling factor of mean vector
    for i in range(vectors_np.shape[0]):
        # Calculate scaling factor using dot product
        scaling_factor = np.dot(vectors_np[i], mean_vector) / np.dot(mean_vector, mean_vector)
        scaling_factors.append(scaling_factor)
        mean_scaled[i] = scaling_factor * mean_vector
    
    # Perform PCA reconstruction
    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver,
        random_state=random_state
    )
    
    pca_transformed = pca.fit_transform(vectors_np)
    print('pca.explained_variance_ratio_', pca.explained_variance_ratio_)
    pca_reconstructed = pca.inverse_transform(pca_transformed)
    
    # Convert reconstructions back to torch tensors
    processed_pca = torch.from_numpy(pca_reconstructed).to(original_directions.device, original_directions.dtype)
    processed_mean = torch.from_numpy(mean_scaled).to(original_directions.device, original_directions.dtype)
    
    # Calculate reconstruction errors
    pca_error = np.mean(np.square(vectors_np - pca_reconstructed))
    mean_error = np.mean(np.square(vectors_np - mean_scaled))
    
    print(f"\nReconstruction Errors:")
    print(f"PCA Error: {pca_error:.6f}")
    print(f"Mean-Scaled Error: {mean_error:.6f}")
    
    # Calculate correlation between original and reconstructions
    pca_corr = np.corrcoef(vectors_np.flatten(), pca_reconstructed.flatten())[0,1]
    mean_corr = np.corrcoef(vectors_np.flatten(), mean_scaled.flatten())[0,1]
    pca_mean_corr = np.corrcoef(pca_reconstructed.flatten(), mean_scaled.flatten())[0,1]
    
    print(f"\nCorrelations with Original:")
    print(f"PCA Correlation: {pca_corr:.6f}")
    print(f"Mean-Scaled Correlation: {mean_corr:.6f}")
    print(f"PCA to Mean Correlation: {pca_mean_corr:.6f}")
    
    # Plot comparison of scaling factors vs first PCA component
    plt.figure(figsize=(10, 6))
    plt.scatter(scaling_factors, pca_transformed[:, 0], alpha=0.5)
    plt.xlabel('Mean Vector Scaling Factors')
    plt.ylabel('First PCA Component')
    plt.title('Comparison of Mean Scaling vs First PCA Component')
    plt.grid(True)
    plt.show()
    
    # Return both reconstructions for comparison
    return processed_pca#, processed_mean, scaling_factors

def create_pca_caa_steering_vector_with_stats(model, sae, pos_tokens, neg_tokens, layer, n_components=1, whiten=False, svd_solver='auto'):
    """TODO: tidy up
    Extract and process activation differences using CAA. Use PCA to extract steering direction from a set of directions"""

    # Get activations for positive and negative sequences
    _, _, pos_cache = model.forward(pos_tokens, cache_activations_at=[layer])
    _, _, neg_cache = model.forward(neg_tokens, cache_activations_at=[layer])
    
    pos_acts = pos_cache[layer][0]
    neg_acts = neg_cache[layer][0]

    if len(pos_acts) > len(neg_acts):
        print('Warning!!! Pos acts longer than neg acts')
        diff = len(pos_acts) - len(neg_acts)
        pos_acts = pos_acts[:-diff]
    elif len(neg_acts) > len(pos_acts):
        print('Warning!!! Neg acts longer than pos acts')
        diff = len(neg_acts) - len(pos_acts)
        neg_acts = neg_acts[:-diff]

    # Find the first position where the tokens differ
    diff_start = 0
    for i in range(len(pos_tokens[0])):
        if pos_tokens[0][i] != neg_tokens[0][i]:
            diff_start = i
            break

    # Find center point between positive and negative activations
    center = (pos_acts + neg_acts) / 2
    
    # Get vectors from center to positive point
    pos_directions = pos_acts - center

    # Only consider the positions after the first difference
    pos_directions = pos_directions[diff_start:]

    processed_caa = pca_directions(
        pos_directions,
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver
    )

    # Average across all token positions (dim=0)
    processed_caa = processed_caa.mean(dim=0, keepdim=True)

    encoded_caa = sae.encode(processed_caa)
    #feature_encoding = torch.zeros(sae.d_hidden, device=sae.device, dtype=torch.bfloat16)
    #feature_encoding[pirate_feature_index] = 1.0
    decoded_caa = sae.decode(encoded_caa)

        # Calculate cosine similarity for each token position
    print("\nToken-wise cosine similarities:")
    print("-" * 80)
    
    # Decode positive tokens
    pos_decoded = model.tokenizer.batch_decode(pos_tokens[0])
    neg_decoded = model.tokenizer.batch_decode(neg_tokens[0])
    
    
    results = []
    # For each token position
    for i in range(len(pos_acts)):
        # Calculate cosine similarity between this position and the mean vector
        pos_sim = torch.nn.functional.cosine_similarity(
            pos_acts[i].reshape(1, -1),
            processed_caa,
            dim=1
        ).item()

        neg_sim = torch.nn.functional.cosine_similarity(
            neg_acts[i].reshape(1, -1),
            processed_caa,
            dim=1
        ).item()

        center_sim = torch.nn.functional.cosine_similarity(
            center[i].reshape(1, -1),
            processed_caa,
            dim=1
        ).item()

        if i >= diff_start:
            pos_dir_sim = torch.nn.functional.cosine_similarity(
                pos_directions[i-diff_start].reshape(1, -1),
                processed_caa,
                dim=1
            ).item()
        else:
            pos_dir_sim = 0

        results.append({
            'pos_sim': pos_sim,
            'neg_sim': neg_sim,
            'center_sim': center_sim,
            'pos_dir_sim': pos_dir_sim,
        })

    for i, result in enumerate(results):
        pos_sim = result['pos_sim']
        neg_sim = result['neg_sim']
        center_sim = result['center_sim']
        pos_dir_sim = result['pos_dir_sim']
        # Print token and its similarity
        print(f"Position {i:2d} | Token: {repr(pos_decoded[i]):20s} | Cos Sim: {pos_sim:6.3f}")
        if i < len(neg_decoded):
            print(f"            | Token: {repr(neg_decoded[i]):20s} | Cos Sim: {neg_sim:6.3f}")
        print(f"            | Token: {repr(pos_decoded[i]):20s} | Cos Sim: {center_sim:6.3f}")
        print(f"            | Token: {repr(pos_decoded[i]):20s} | Cos Sim: {pos_dir_sim:6.3f}")
        print("-" * 80)


    processed_caa = processed_caa / torch.norm(processed_caa)

    return processed_caa

def create_pca_caa_steering_vector(model, pos_tokens, neg_tokens, layer, n_components=1, whiten=False, svd_solver='auto'):
    """TODO: tidy up
    Extract and process activation differences using CAA. Use PCA to extract steering direction from a set of directions"""

    # Get activations for positive and negative sequences
    _, _, pos_cache = model.forward(pos_tokens, cache_activations_at=[layer])
    _, _, neg_cache = model.forward(neg_tokens, cache_activations_at=[layer])
    
    pos_acts = pos_cache[layer][0]
    neg_acts = neg_cache[layer][0]

    if len(pos_acts) > len(neg_acts):
        print('Warning!!! Pos acts longer than neg acts')
        diff = len(pos_acts) - len(neg_acts)
        pos_acts = pos_acts[:-diff]
    elif len(neg_acts) > len(pos_acts):
        print('Warning!!! Neg acts longer than pos acts')
        diff = len(neg_acts) - len(pos_acts)
        neg_acts = neg_acts[:-diff]

    # Find the first position where the tokens differ
    diff_start = 0
    for i in range(len(pos_tokens[0])):
        if pos_tokens[0][i] != neg_tokens[0][i]:
            diff_start = i
            break

    # Find center point between positive and negative activations
    center = (pos_acts + neg_acts) / 2
    
    # Get vectors from center to positive point
    pos_directions = pos_acts - center

    # Only consider the positions after the first difference
    pos_directions = pos_directions[diff_start:]

    processed_caa = pca_directions(
        pos_directions,
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver
    )

    # Average across all token positions (dim=0)
    processed_caa = processed_caa.mean(dim=0, keepdim=True)

    processed_caa = processed_caa / torch.norm(processed_caa)

    return processed_caa

def pca_experiment(model, sae, client, model_name, positive_tokens, neutral_tokens, layer, n_components=1, whiten=True, svd_solver='full'):
    pirate_feature_index = 58644
    # pirate_feature_strength = 12.0
    # pirate_feature = {pirate_feature_index: pirate_feature_strength}

       # Create aggregate steering vector
    original_caa_vector = create_mean_caa_steering_vector(model, positive_tokens, neutral_tokens, layer)
    pca_caa_vector = create_pca_caa_steering_vector(model, positive_tokens, neutral_tokens, layer, n_components, whiten, svd_solver)

    # sae_vector = create_sae_steering_vector(sae, pirate_feature)
    # sae_feature = create_sae_steering_vector_latents(sae, pirate_feature)

    # get latents for caa vector
    pca_caa_features = sae.encode(pca_caa_vector)
    pca_caa_features = pca_caa_features / torch.norm(pca_caa_features)
    # pca_caa_reconstructed = sae.decode(pca_caa_features)
    
    original_caa_features = sae.encode(original_caa_vector)
    original_caa_features = original_caa_features / torch.norm(original_caa_features)
    # original_caa_reconstructed = sae.decode(original_caa_features)

    # # Calculate errors using normalized vectors
    # original_caa_error = original_caa_vector - original_caa_reconstructed
    # pca_caa_error = original_caa_vector - pca_caa_reconstructed

    # print('original norm', torch.norm(original_caa_vector).item())
    # print('pca norm', torch.norm(pca_caa_vector).item())
    # print('original reconstructed norm', torch.norm(original_caa_reconstructed).item())
    # print('pca reconstructed norm', torch.norm(pca_caa_reconstructed).item())
    # print('original error norm', torch.norm(original_caa_error).item())
    # print('pca error norm', torch.norm(pca_caa_error).item())

    # # Calculate Explained Variance (EV) metric
    # original_ev = 1 - (torch.norm(original_caa_error) ** 2) / (torch.norm(original_caa_vector) ** 2)
    # pca_ev = 1 - (torch.norm(pca_caa_error) ** 2) / (torch.norm(pca_caa_vector) ** 2)

    # print(f"\nPCA CAA Explained Variance: {pca_ev:.1%}")
    # print(f"Original CAA Explained Variance: {original_ev:.1%}")

    # pca_similarity = compare_steering_vectors(pca_caa_features, sae_feature)
    # original_similarity = compare_steering_vectors(original_caa_features, sae_feature)
    # caa_similarity = compare_steering_vectors(pca_caa_features, original_caa_features)
    # model_caa_similarity = compare_steering_vectors(pca_caa_vector, original_caa_vector)
    # print(f"\nCosine similarity between PCA CAA and SAE vectors in SAE latent space: {pca_similarity:.3f}")
    # print(f"\nCosine similarity between original CAA and SAE vectors in SAE latent space: {original_similarity:.3f}")
    # print(f"\nCosine similarity between original CAA and PCA CAA vectors in SAE latent space: {caa_similarity:.3f}")
    # print(f"\nCosine similarity between original CAA and PCA CAA vectors in model latent space: {model_caa_similarity:.3f}")


    metrics_pca_vs_orig = visualize_vector_comparison(
        pca_caa_vector.detach().float(),
        original_caa_vector.detach().float(),
        "PCA vs Original CAA in Activation Space"
    )
    
    # metrics_pca_vs_sae = visualize_vector_comparison(
    #     pca_caa_vector.detach().float(),
    #     sae_vector.detach().float(),
    #     "PCA CAA vs SAE in Activation Space"
    # )

    # metrics_pca_vs_orig = visualize_vector_comparison(
    #     pca_caa_features.detach().float(),
    #     original_caa_features.detach().float(),
    #     "PCA vs Original CAA in SAE Latent Space"
    # )
    
    # metrics_pca_vs_sae = visualize_vector_comparison(
    #     pca_caa_features.detach().float(),
    #     sae_feature.detach().float(),
    #     "PCA CAA vs SAE in SAE Latent Space"
    # )
    
    
    # Get features as numpy arrays
    pca_features = pca_caa_features[0].detach().float().cpu().numpy()
    original_features = original_caa_features[0].detach().float().cpu().numpy()

    # Create combined dictionary of all feature values
    feature_dict = {}
    for idx in range(len(pca_features)):
        if pca_features[idx] != 0 or original_features[idx] != 0:
            feature_dict[idx] = {
                'pca': pca_features[idx],
                'original': original_features[idx],
                'combined': abs(pca_features[idx]) + abs(original_features[idx])
            }

    # Sort by combined absolute values
    sorted_features = sorted(
        feature_dict.items(), 
        key=lambda x: x[1]['combined'], 
        reverse=True
    )

    # Create histogram
    plt.figure(figsize=(12, 6))
    
    # Get non-zero values for both
    pca_nonzero = pca_features[pca_features != 0]
    original_nonzero = original_features[original_features != 0]
    
    # Find common range for bins
    min_val = min(pca_nonzero.min(), original_nonzero.min())
    max_val = max(pca_nonzero.max(), original_nonzero.max())
    bins = np.linspace(min_val, max_val, 50)
    
    # Plot histograms
    plt.hist(pca_nonzero, bins=bins, alpha=0.5, label='PCA', density=True)
    plt.hist(original_nonzero, bins=bins, alpha=0.5, label='Original', density=True)
    
    plt.title('Distribution of Non-Zero Latent Values for Steering Vectors in SAE Latent Space')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add vertical lines for pirate feature if non-zero
    if pca_features[pirate_feature_index] != 0:
        plt.axvline(x=pca_features[pirate_feature_index], color='red', 
                   linestyle='--', label=f'Pirate (PCA): {pca_features[pirate_feature_index]:.3f}')
    if original_features[pirate_feature_index] != 0:
        plt.axvline(x=original_features[pirate_feature_index], color='darkred',
                   linestyle=':', label=f'Pirate (Original): {original_features[pirate_feature_index]:.3f}')
    plt.legend()
    plt.show()

    # Get all feature descriptions in one API call
    if sorted_features:
        feature_indices = [idx for idx, _ in sorted_features]
        feature_descriptions = client.features.lookup(feature_indices, model_name)
        print("\nFeature descriptions (sorted by combined magnitude):")
        print(f"{'Index':<8} {'PCA':<10} {'Original':<10} {'Delta (%)':<10} Description")
        print("-" * 80)
        
        for idx, values in sorted_features:
            feature = feature_descriptions.get(idx)
            label = feature.label if feature else "<Redacted due to sensitivity>"
            
            print(f"{idx:<8} {values['pca']:<10.3f} {values['original']:<10.3f} "
                  f"{100 * abs((values['original'] - values['pca']) / values['original']):<10.0f} {label}")
            
            if idx == pirate_feature_index:
                print("^^ Pirate feature! ^^")


    return pca_caa_vector, original_caa_vector

def compare_steering_vectors(caa_vector, sae_vector):
    """TODO: can I remove this?
    Compare CAA steering vector with SAE-projected steering"""

    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        caa_vector.reshape(-1),
        sae_vector.reshape(-1),
        dim=0
    )
    
    return similarity.item()

def compare_vectors(v1, v2):
    """TODO: tidy up
    Compare two vectors using multiple metrics"""
    # Ensure vectors are on CPU and in float32
    v1 = v1.detach().float().cpu()
    v2 = v2.detach().float().cpu()
    
    # 1. Cosine Similarity (already implemented)
    cosine_sim = torch.nn.functional.cosine_similarity(
        v1.reshape(-1), 
        v2.reshape(-1), 
        dim=0
    ).item()
    
    # 2. Euclidean Distance
    euclidean_dist = torch.norm(v1 - v2).item()
    
    # 3. Manhattan Distance (L1 norm)
    manhattan_dist = torch.norm(v1 - v2, p=1).item()
    
    # 4. Pearson Correlation
    v1_centered = v1 - v1.mean()
    v2_centered = v2 - v2.mean()
    pearson_corr = torch.sum(v1_centered * v2_centered) / (
        torch.norm(v1_centered) * torch.norm(v2_centered)
    ).item()
    
    # 5. Sparsity comparison
    sparsity1 = (v1 == 0).float().mean().item()
    sparsity2 = (v2 == 0).float().mean().item()
    
    # 6. Top-k overlap
    def get_top_k_indices(v, k=100):
        return set(torch.topk(v.abs().reshape(-1), k).indices.tolist())
    
    top_k = 100
    top_k_overlap = len(
        get_top_k_indices(v1, top_k) & get_top_k_indices(v2, top_k)
    ) / top_k
    
    return {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'manhattan_distance': manhattan_dist,
        'pearson_correlation': pearson_corr,
        'sparsity_v1': sparsity1,
        'sparsity_v2': sparsity2,
        'top_k_overlap': top_k_overlap
    }

def visualize_vector_comparison(v1, v2, title="Vector Comparison"):
    """TODO: tidy up
    Create visualizations comparing two vectors"""
    metrics = compare_vectors(v1, v2)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot of values
    ax1.scatter(
        v1.reshape(-1).cpu().numpy(),
        v2.reshape(-1).cpu().numpy(),
        alpha=0.1
    )
    ax1.set_xlabel('Vector 1 Values')
    ax1.set_ylabel('Vector 2 Values')
    ax1.set_title('Value Correlation')
    
    # 2. Distribution comparison
    ax2.hist(
        v1.reshape(-1).cpu().numpy(),
        bins=50,
        alpha=0.5,
        label='Vector 1',
        density=True
    )
    ax2.hist(
        v2.reshape(-1).cpu().numpy(),
        bins=50,
        alpha=0.5,
        label='Vector 2',
        density=True
    )
    ax2.set_title('Value Distributions')
    ax2.legend()
    
    # 3. Absolute difference distribution
    diff = (v1 - v2).abs().cpu().numpy()
    ax3.hist(diff.reshape(-1), bins=50)
    ax3.set_title('Absolute Differences')
    
    # 4. Metrics text
    ax4.axis('off')
    metrics_text = '\n'.join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    ax4.text(0.1, 0.9, metrics_text, fontsize=12, verticalalignment='top')
    
    plt.suptitle(title)
    plt.tight_layout()
    return metrics


def plot_feature_distribution(features, model_name, goodfire_client, target_feature_index, k=0):
    # Find all features with significant activation
    significant_indices = np.where(np.abs(features) > 0)[0]
    significant_values = features[significant_indices]

    # Sort by absolute value
    sorted_order = np.argsort(np.abs(significant_values))[::-1]  # Descending order
    sorted_indices = significant_indices[sorted_order]
    sorted_values = significant_values[sorted_order]

    print(f"Number of non-zero features: {len(sorted_indices)}")

    if k > 0:
        sorted_indices = sorted_indices[:k]
        sorted_values = sorted_values[:k]

    # Get all feature descriptions in one API call
    if len(sorted_indices) > 0:
        feature_descriptions = goodfire_client.features.lookup(sorted_indices.tolist(), model_name)
        print("\nFeature descriptions:")
        for idx, value in zip(sorted_indices, sorted_values):
            feature = feature_descriptions.get(idx)
            if feature is None:
                label = "<Redacted due to sensitivity>"
            else:
                label = feature.label
            print(f"\nFeature {idx} ({value:.3f}): {label}")
            if idx == target_feature_index:
                print(f"Target feature!")

    # Filter out zero values
    nonzero_features = features[features != 0]
    # count the number of inf/non inf features, then filter them out
    inf_count = np.sum(np.isinf(nonzero_features))
    print(f"Number of inf features: {inf_count}")
    nonzero_features = nonzero_features[~np.isinf(nonzero_features)]

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(nonzero_features, bins=50, density=True)
    plt.title('Distribution of Non-Zero CAA Feature Values')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    # Add vertical line for pirate feature if it's non-zero
    target_value = features[target_feature_index]
    if target_value != 0:
        plt.axvline(x=target_value, color='r', linestyle='--', label=f'Target Feature ({target_value:.3f})')
        plt.legend()

    plt.show()


def template_and_tokenize_for_prefix(model, prompt, user_prompt, prefix):
    return model.tokenizer.apply_chat_template(
            [   {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": prefix}],
            return_tensors="pt",
            continue_final_message=True
        )


def get_contrastive_pairs_from_prefixes(model, prefixes_path, limit_size=0):
    positive_prompt = 'The assistant should talk like a pirate.',
    neutral_prompt = 'The assistant should act normally.'
  
    # Load prefixes from JSON
    with open(prefixes_path) as f:
        prefixes = json.load(f)
    
    if limit_size:
        prefixes = prefixes[:limit_size]

    results = [] 
    user_prompt = "Hello, how are you?"

    for prefix in prefixes:
            positive_tokens = template_and_tokenize_for_prefix(model, positive_prompt, user_prompt, prefix)
            neutral_tokens = template_and_tokenize_for_prefix(model, neutral_prompt, user_prompt, prefix)
            
            results.append({
                'prefix': prefix,
                'positive_tokens': positive_tokens,    
                'neutral_tokens': neutral_tokens
            })
            
            # Optional: Print progress
            print(f"Processed {len(results)}/{len(prefixes)} prefixes", end='\r')
    print("\nDone!")
    return results


def extract_caa_vector_from_pairs(model, all_pairs, layer):
    """Extract and aggregate CAA vectors from multiple pairs"""
    all_vectors = []
    
    for pair in all_pairs:
        vector = extract_caa_direction_from_pair(model, pair['positive_tokens'], pair['neutral_tokens'], layer)
        all_vectors.append(vector)
    
    # Stack all vectors and average them
    stacked_vectors = torch.stack(all_vectors)
    aggregate_vector = stacked_vectors.mean(dim=0)
    # Normalize the aggregate vector
    aggregate_vector = aggregate_vector / torch.norm(aggregate_vector)
    
    return aggregate_vector


def extract_caa_direction_from_pair(model, pos_tokens, neg_tokens, layer):
    """Extract and process activation differences using CAA"""
    
    # Get activations for positive and negative sequences
    _, _, pos_cache = model.forward(pos_tokens, cache_activations_at=[layer])
    _, _, neg_cache = model.forward(neg_tokens, cache_activations_at=[layer])
    
    pos_acts = pos_cache[layer][0][-1]
    neg_acts = neg_cache[layer][0][-1]

    # Find center point between positive and negative activations
    center = (pos_acts + neg_acts) / 2
    
    # Get vectors from center to positive/negative points
    pos_vector = pos_acts - center
    #neg_vector = neg_acts - center
    
    # Use positive direction and normalize
    if torch.norm(pos_vector):
        pos_vector = pos_vector / torch.norm(pos_vector)

    return pos_vector