"""Gemma models and their associated SAE configurations (residual stream only)."""

gemma_models_and_saes = {}

# Add all layers for gemma-2-2b-res-16k (layers 0-25)
for layer in range(26):
    gemma_models_and_saes[f"google/gemma-2-2b-res-16k-layer-{layer}"] = {
        "model_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": f"layer_{layer}/width_16k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Add all layers for gemma-2-2b-res-65k (layers 0-25)
for layer in range(26):
    gemma_models_and_saes[f"google/gemma-2-2b-res-65k-layer-{layer}"] = {
        "model_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": f"layer_{layer}/width_65k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-2b IT - Residual Stream, 16k width (layers 0-25)
# NOTE: Using PT SAE because IT SAE doesn't exist for 2b
for layer in range(26):
    gemma_models_and_saes[f"google/gemma-2-2b-it-res-16k-layer-{layer}"] = {
        "model_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",  # PT SAE (no IT SAE available)
        "sae_id": f"layer_{layer}/width_16k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-2b IT - Residual Stream, 65k width (layers 0-25)
# NOTE: Using PT SAE because IT SAE doesn't exist for 2b
for layer in range(26):
    gemma_models_and_saes[f"google/gemma-2-2b-it-res-65k-layer-{layer}"] = {
        "model_id": "google/gemma-2-2b-it",
        "sae_release": "gemma-scope-2b-pt-res-canonical",  # PT SAE (no IT SAE available)
        "sae_id": f"layer_{layer}/width_65k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-9b PT - Residual Stream, 16k width (layers 0-41)
for layer in range(42):
    gemma_models_and_saes[f"google/gemma-2-9b-res-16k-layer-{layer}"] = {
        "model_id": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "sae_id": f"layer_{layer}/width_16k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-9b PT - Residual Stream, 65k width (layers 0-41)
for layer in range(42):
    gemma_models_and_saes[f"google/gemma-2-9b-res-65k-layer-{layer}"] = {
        "model_id": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "sae_id": f"layer_{layer}/width_65k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-9b PT - Residual Stream, 131k width (layers 0-41)
for layer in range(42):
    gemma_models_and_saes[f"google/gemma-2-9b-res-131k-layer-{layer}"] = {
        "model_id": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-pt-res-canonical",
        "sae_id": f"layer_{layer}/width_131k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-9b IT - Residual Stream, 16k width (layers 0-41)
for layer in range(42):
    gemma_models_and_saes[f"google/gemma-2-9b-it-res-16k-layer-{layer}"] = {
        "model_id": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-it-res-canonical",
        "sae_id": f"layer_{layer}/width_16k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-9b IT - Residual Stream, 65k width (layers 0-41)
for layer in range(42):
    gemma_models_and_saes[f"google/gemma-2-9b-it-res-65k-layer-{layer}"] = {
        "model_id": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-it-res-canonical",
        "sae_id": f"layer_{layer}/width_65k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-9b IT - Residual Stream, 131k width (layers 0-41)
for layer in range(42):
    gemma_models_and_saes[f"google/gemma-2-9b-it-res-131k-layer-{layer}"] = {
        "model_id": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-it-res-canonical",
        "sae_id": f"layer_{layer}/width_131k/canonical",
        "steering_layer": layer,
        "feature_layer": layer,
        "quantization": None,
    }

# Gemma-2-27b PT - Residual Stream, 131k width
# Available at layers: 10, 22, 34
gemma_models_and_saes.update({
    "google/gemma-2-27b-res-131k-layer-10": {
        "model_id": "google/gemma-2-27b-it",
        "sae_release": "gemma-scope-27b-pt-res-canonical",
        "sae_id": "layer_10/width_131k/canonical",
        "steering_layer": 10,
        "feature_layer": 10,
        "quantization": None,
    },
    "google/gemma-2-27b-res-131k-layer-22": {
        "model_id": "google/gemma-2-27b-it",
        "sae_release": "gemma-scope-27b-pt-res-canonical",
        "sae_id": "layer_22/width_131k/canonical",
        "steering_layer": 22,
        "feature_layer": 22,
        "quantization": None,
    },
    "google/gemma-2-27b-res-131k-layer-34": {
        "model_id": "google/gemma-2-27b-it",
        "sae_release": "gemma-scope-27b-pt-res-canonical",
        "sae_id": "layer_34/width_131k/canonical",
        "steering_layer": 34,
        "feature_layer": 34,
        "quantization": None,
    },

    # Gemma-2-27b IT - Residual Stream, 131k width
    # Available at layers: 10, 22, 34
    # NOTE: Using PT SAE because IT SAE doesn't exist for 27b
    "google/gemma-2-27b-it-res-131k-layer-10": {
        "model_id": "google/gemma-2-27b-it",
        "sae_release": "gemma-scope-27b-pt-res-canonical",  # PT SAE (no IT SAE available)
        "sae_id": "layer_10/width_131k/canonical",
        "steering_layer": 10,
        "feature_layer": 10,
        "quantization": None,
    },
    "google/gemma-2-27b-it-res-131k-layer-22": {
        "model_id": "google/gemma-2-27b-it",
        "sae_release": "gemma-scope-27b-pt-res-canonical",  # PT SAE (no IT SAE available)
        "sae_id": "layer_22/width_131k/canonical",
        "steering_layer": 22,
        "feature_layer": 22,
        "quantization": None,
    },
    "google/gemma-2-27b-it-res-131k-layer-34": {
        "model_id": "google/gemma-2-27b-it",
        "sae_release": "gemma-scope-27b-pt-res-canonical",  # PT SAE (no IT SAE available)
        "sae_id": "layer_34/width_131k/canonical",
        "steering_layer": 34,
        "feature_layer": 34,
        "quantization": None,
    },
})
