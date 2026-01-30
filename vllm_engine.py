"""vLLM engine wrapper for feature steering experiments."""

import asyncio
import uuid
from typing import List, Dict, Optional

import torch
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokenInputs, InterventionInputs
from vllm.model_executor.models.llama_models_and_saes import llama_models_and_saes
from gemma_models_and_saes import gemma_models_and_saes


# Model-specific repetition penalties based on observed degradation rates
# Higher penalties for models with more repetition issues under steering
# Llama 70B: 39% baseline degradation -> needs aggressive penalty
# Llama 8B: 24% -> moderate penalty
# Gemma models: 6-18% -> lighter penalties
MODEL_REPETITION_PENALTIES = {
    "70b": 1.2,    # Llama 70B - high degradation, tested
    "8b": 1.15,   # Llama 8B - moderate degradation
    "27b": 1.1,   # Gemma 27B - lower degradation
    "9b": 1.1,    # Gemma 9B - lower degradation
    "2b": 1.1,    # Gemma 2B - bumped from 1.05 to reduce degradation
}
DEFAULT_REPETITION_PENALTY = 1.1


def get_repetition_penalty(model_path: str) -> float:
    """Get the appropriate repetition penalty for a model based on its size."""
    model_lower = model_path.lower()
    for size_key, penalty in MODEL_REPETITION_PENALTIES.items():
        if size_key in model_lower:
            return penalty
    return DEFAULT_REPETITION_PENALTY


class VLLMSteeringEngine:
    """Wrapper for vLLM AsyncLLMEngine with feature steering support."""

    def __init__(
        self,
        model_str: str,
        gpu_memory_utilization: float = 0.90,
        base_model_for_sae: Optional[str] = None,
        repetition_penalty: Optional[float] = None,
    ):
        """
        Initialize the vLLM engine with SAE support.

        Args:
            model_str: Model identifier (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
                      or path to local model (e.g., "finetuning/outputs-lora-8b-self-correction/run-1")
            gpu_memory_utilization: GPU memory utilization ratio
            base_model_for_sae: If model_str is a local path, specify which HuggingFace model's
                               SAE configuration to use (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
            repetition_penalty: Override the default repetition penalty for this model.
                               If None, uses the model-specific default from get_repetition_penalty().
        """
        self.model_str = model_str
        self._repetition_penalty_override = repetition_penalty

        # Combine both model configs
        all_models = {**llama_models_and_saes, **gemma_models_and_saes}

        # Check if model_str is in the predefined configs or is a local path
        if model_str in all_models:
            # Use the predefined config for HuggingFace models
            self.model_config = all_models[model_str]
            self.model_path = self.model_config["model_id"]
        else:
            # Treat as local model path
            if base_model_for_sae is None:
                raise ValueError(
                    f"Model '{model_str}' not found in llama_models_and_saes or gemma_models_and_saes. "
                    "For local models, you must specify base_model_for_sae to indicate "
                    "which model's SAE configuration to use."
                )
            if base_model_for_sae not in all_models:
                raise ValueError(
                    f"base_model_for_sae '{base_model_for_sae}' not found in llama_models_and_saes or gemma_models_and_saes"
                )
            # Use SAE config from base model, but load the local model
            self.model_config = all_models[base_model_for_sae]
            self.model_path = model_str  # Use local path for model loading

        self.engine = None
        self.tokenizer = None
        self.gpu_memory_utilization = gpu_memory_utilization

    async def initialize(self):
        """Initialize the async engine and tokenizer."""
        # Use FlashInfer backend for Gemma models (supports tanh softcapping)
        import os
        if "gemma" in self.model_path.lower():
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

        # Build engine args based on model type (Llama vs Gemma)
        engine_args = {
            "model": self.model_path,
            "dtype": torch.bfloat16,
            "enforce_eager": True,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": 4096,
            "tensor_parallel_size": torch.cuda.device_count(),
            "steering_layer": self.model_config["steering_layer"],
            "feature_layer": self.model_config["feature_layer"],
            "quantization": self.model_config["quantization"],
            "enable_prefix_caching": False,  # Disable prefix caching for consistent steering
            "steering_scale_factor": 0,
        }

        # Add SAE parameters based on model type
        if "sae_release" in self.model_config:
            # Gemma-style SAE config (using SAELens)
            engine_args["sae_release"] = self.model_config["sae_release"]
            engine_args["sae_id"] = self.model_config["sae_id"]
        else:
            # Llama-style SAE config (using custom SAE)
            engine_args["sae_name"] = self.model_config["sae_id"]
            engine_args["sae_filepath"] = self.model_config["sae_filepath"]
            engine_args["hidden_size"] = self.model_config["d_model"]
            engine_args["sae_expansion_factor"] = self.model_config["expansion_factor"]

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))
        self.tokenizer = await self.engine.get_tokenizer()

        # Set Gemma chat template if needed
        if "gemma" in self.model_path.lower() and not self.tokenizer.chat_template:
            # Gemma 2 chat template
            self.tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        feature_interventions: Optional[List[Dict[str, float]]] = None,
        temperature: float = 0.6,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate text with optional feature interventions.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            feature_interventions: List of dicts with 'feature_id' and 'strength' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            seed: Random seed for generation

        Returns:
            Generated text string
        """
        # Apply chat template and get token IDs
        prompt_token_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Add assistant header tokens for Llama models only
        if 'llama' in self.model_path.lower():
            prompt_token_ids.extend([128000, 128006, 78191, 128007])

        # Create token inputs
        token_inputs = TokenInputs(prompt_token_ids=prompt_token_ids, prompt=messages)

        # Create interventions if provided
        interventions = None
        if feature_interventions:
            interventions = InterventionInputs(intervention=feature_interventions)

        # Set up sampling parameters with model-specific repetition penalty
        rep_penalty = self._repetition_penalty_override if self._repetition_penalty_override is not None else get_repetition_penalty(self.model_path)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=rep_penalty,
            seed=seed if seed is not None else None,
        )

        # Generate
        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(
            prompt=token_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
            interventions=interventions,
            is_feature_decode=False,
        )

        # Wait for final output
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        # Extract text
        return final_output.outputs[0].text

    async def generate_with_conversation(
        self,
        conversation: List[Dict[str, str]],
        feature_interventions: Optional[List[Dict[str, float]]] = None,
        temperature: float = 0.6,
        max_tokens: int = 512,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate continuation of a conversation with feature interventions.

        This is the main method used in the experiment - it takes a full conversation
        history and generates the next assistant response.

        Args:
            conversation: Full conversation history as list of message dicts
            feature_interventions: Optional feature steering interventions
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            seed: Random seed

        Returns:
            Generated assistant response text
        """
        return await self.generate(
            messages=conversation,
            feature_interventions=feature_interventions,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
