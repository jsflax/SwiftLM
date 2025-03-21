import logging
import os
from coremltools.models import MLModel
from coremltools.models.neural_network.printer import print_network_spec


import torch
import torch.nn as nn
import coremltools as ct
import coremltools.optimize as cto
import numpy as np
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights, \
    OpLinearQuantizerConfig, linear_quantize_weights

from transformers.models.auto import AutoTokenizer
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.models.auto import AutoConfig

from modeling_llama import StatefulLlamaForCausalLM
from modeling_mistral import StatefulMistralForCausalLM
logging.getLogger("coremltools").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# MODEL_ID: str = "JesperSqv/Llama-3.2-1B-Instruct-Financial-RAG"
MODEL_ID: str = 'mistralai/Mistral-Small-24B-Instruct-2501'
METADATA_TOKENIZER: str = "co.huggingface.exporters.name"
BASE_NAME = os.path.basename(MODEL_ID)

max_context_size: int = 8192
config = AutoConfig.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if config.architectures[0] == "MistralForCausalLM":
    is_mistral = True
else:
    is_mistral = False

if config.architectures[0] == "MistralForCausalLM":
    CausalLM = StatefulMistralForCausalLM
elif config.architectures[0] == "LlamaForCausalLM":
    CausalLM = StatefulLlamaForCausalLM
else:
    print('Unsupported model type. LLMKit currently only supports Mistral and Llama models.')
    exit(1)

tokenizer.save_pretrained(f"models/{BASE_NAME}_tokenizer")
# config = get_tokenizer_config(MODEL_ID)
# print(config)
# auto_config = AutoConfig.from_pretrained(MODEL_ID)
def test_model_output(mlmodel: MLModel,
                      input_ids_pt,
                      attention_mask_pt,
                      use_fixed_size: bool,
                      use_state: bool):
    eos_token_id = tokenizer.eos_token_id
    generated_ids = input_ids_pt[0].tolist()  # (1, prompt_length) -> prompt_length tokens

    if use_fixed_size:
        # Number of active tokens
        used_tokens = (attention_mask_pt[0] == 1).sum().item()
    else:
        # Just use the prompt length
        used_tokens = input_ids_pt.shape[1]

    if use_state:
        state = mlmodel.make_state()
    else:
        state = None

    decoded_text_so_far = tokenizer.decode(generated_ids[:used_tokens])

    for i in range(100):
        input_ids_np = np.array([generated_ids], dtype=np.int32)

        if use_fixed_size:
            # Fixed sized mask
            attention_mask_np = np.zeros((1, max_context_size), dtype=np.int32)
            attention_mask_np[0, :used_tokens] = 1
        elif use_state:
            # Causal mask (lower-triangular), same shape each step
            # Make sure dtype matches what model expects (float16)
            attention_mask_np = _generate_causal_mask(used_tokens)
        else:
            # Flexible size without state: just use a 2D mask of shape (1, used_tokens)
            attention_mask_np = np.ones((1, used_tokens), dtype=np.int32)

        # Run inference
        if use_state:
            predictions = mlmodel.predict({
                "inputIds": input_ids_np,
                "causalMask": attention_mask_np
            }, state=state)
        else:
            predictions = mlmodel.predict({
                "inputIds": input_ids_np,
                "attentionMask": attention_mask_np
            })

        logits = predictions["logits"]  # Check the shape of logits

        # Handle indexing based on how model returns logits
        if use_state:
            # If incremental decoding: logits shape might be (1,1,vocab_size)
            last_token_logits = logits[0, -1, :]
        else:
            # Full sequence decoding: logits shape might be (1, used_tokens, vocab_size)
            last_token_logits = logits[0, used_tokens - 1, :]

        # Choose next token (greedy)
        next_token_id = int(np.argmax(last_token_logits))

        # Append next token
        if use_fixed_size:
            if used_tokens < max_context_size:
                generated_ids[used_tokens] = next_token_id
            else:
                print("Reached max context size.")
                break
        else:
            generated_ids.append(next_token_id)

        used_tokens += 1

        # Stop if EOS
        if next_token_id == eos_token_id:
            break

        full_text = tokenizer.decode(generated_ids[:used_tokens], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        new_text = full_text[len(decoded_text_so_far):]
        decoded_text_so_far = full_text
        print(new_text, end="", flush=True)

    print("\nFinal output:")
    final_output = tokenizer.decode(generated_ids[:used_tokens], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
    print(final_output)

def convert_model_to_a16w8(mlmodel: MLModel):
    # Create a quantization configuration for weights only (to INT8)
    weight_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",  # or "linear_symmetric" / "linear_lut" etc.
            dtype="int8"  # Set weights to 8-bit integers
        )
    )

    # Apply weight-only quantization
    mlmodel_w8 = cto.coreml.linear_quantize_weights(mlmodel, config=weight_config)

    # The resulting model has W8A16 (since activations remain FP16 from the original model)
    mlmodel_w8.save(f"models/{BASE_NAME}_A16W8.mlpackage")
    print('Converted A16W8')
    return mlmodel_w8

def convert_model_a8w8(mlmodel: MLModel):
    # first palettize the model
    # this will produce an LUT with Float values
    op_config = OpPalettizerConfig(nbits=8)
    config = OptimizationConfig(global_config=op_config)
    mlmodel_palettized = palettize_weights(mlmodel, config)
    mlmodel_palettized.save(f'models/{BASE_NAME}_Fixed_Stateless_Int8.mlpackage')
    # now apply weight quantization on the model,
    # with "joint_compression" set to True.
    # this will result in quantizing the LUT to 8 bits.
    # (granularity must be set to "per-tensor" for this scenario)
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric",
                                        granularity="per_tensor")
    linear_weight_quantize_config = OptimizationConfig(global_config=op_config)

    mlmodel_palettized_with_8bit_lut = linear_quantize_weights(mlmodel_palettized,
                                                               linear_weight_quantize_config,
                                                               joint_compression=True)
    mlmodel_palettized_with_8bit_lut.save(f'models/{BASE_NAME}_A8W8.mlpackage')
    print('Converted to A8W8')
    return mlmodel_palettized_with_8bit_lut

def convert_model_a8w4(mlmodel: MLModel):
    # Create a quantization configuration for weights only (to INT8)
    weight_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",  # or "linear_symmetric" / "linear_lut" etc.
            dtype="int8"  # Set weights to 8-bit integers
        )
    )

    # Apply weight-only quantization
    mlmodel_w8 = cto.coreml.linear_quantize_weights(mlmodel, config=weight_config)

    op_config = OpPalettizerConfig(nbits=4)
    config = OptimizationConfig(global_config=op_config)
    mlmodel_palettized = palettize_weights(mlmodel_w8, config)
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric",
                                        granularity="per_tensor")
    linear_weight_quantize_config = OptimizationConfig(global_config=op_config)
    mlmodel_palettized_with_8bit_lut = linear_quantize_weights(mlmodel_palettized,
                                                               linear_weight_quantize_config,
                                                               joint_compression=True)
    mlmodel_palettized_with_8bit_lut.save(f'models/{BASE_NAME}_A8W4.mlpackage')
    print('Converted to A8W4')
    return mlmodel_palettized_with_8bit_lut

def convert_model_to_int4(mlmodel: MLModel):
    # # Block-wise quantize model weights to int4
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
    mlmodel_int4._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
    mlmodel_int4.save(f"models/{BASE_NAME}_Int4.mlpackage")
    return mlmodel_int4

def _generate_causal_mask(prompt_length: int):
    # Python version of the Swift logic
    maskValues = []
    for i in range(prompt_length):
        for j in range(prompt_length):
            value = 1.0 if j <= i else 0.0
            maskValues.append(value)

    causalMask_np = np.array(maskValues, dtype=np.float16).reshape(1, 1, prompt_length, prompt_length)
    return causalMask_np

def export(use_fixed_size: bool = False,
           use_state: bool = False) -> None:
    # if tokenizer.pad_token_type_id
    templated = tokenizer.apply_chat_template([
        {
            "role": "system",
            "content": "You are an AI assistant. Continue the conversation below."
        },
        {
            "role": "user",
            "content": "How are you doing today?"
        }
    ])
    templated_decoded = tokenizer.decode(templated)
    if is_mistral:
        prompt = """
        <s>[SYSTEM_PROMPT]You are an AI assistant. Continue the conversation below.[/SYSTEM_PROMPT][INST]How are you doing today?[/INST]
        """
    else:
        prompt = """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are an AI assistant.

                Continue the conversation below.<|eot_id|>
                <|start_header_id|>user<|end_header_id|>
                How are you doing today?
                <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    # Tokenize the prompt
    tokenized = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_context_size)
    input_ids_pt = tokenized["input_ids"]  # shape: (1, prompt_length)
    prompt_length = input_ids_pt.shape[1]

    if use_state:
        # input_ids_pt = torch.zeros((1, 2), dtype=torch.int32)
        attention_mask_pt =  torch.from_numpy(_generate_causal_mask(prompt_length)).to(torch.float32) # torch.zeros((1, 1, 2, 5), dtype=torch.float32)
    else:
        attention_mask_pt = tokenized["attention_mask"]  # shape: (1, prompt_length)

    if use_fixed_size:
        if prompt_length < max_context_size:
            pad_length = max_context_size - prompt_length
            pad_ids = torch.full((1, pad_length), tokenizer.pad_token_type_id, dtype=torch.long)
            pad_mask = torch.zeros((1, pad_length), dtype=torch.long)
            # Append pads at the end (right-padding)
            input_ids_pt = torch.cat([input_ids_pt, pad_ids], dim=1)
            attention_mask_pt = torch.cat([attention_mask_pt, pad_mask], dim=1)
        else:
            # If prompt is longer or equal to max_context_size, it’s already set
            input_ids_pt = input_ids_pt[:, :max_context_size]
            attention_mask_pt = attention_mask_pt[:, :max_context_size]

    if os.path.exists(f"models/{BASE_NAME}.mlpackage"):
        mlmodel = ct.models.MLModel(f"models/{BASE_NAME}.mlpackage")
    else:
        # Construct model and tokenizer
        if use_state:
            print('Loading Stateful LM')
            torch_model = CausalLM(MODEL_ID, max_context_size=max_context_size)
        else:
            print('Loading Simple LM')
            torch_model = SimpleLlamaModel(MODEL_ID)
        torch_model.eval()

        # Trace the PyTorch model
        traced_model = torch.jit.trace(torch_model, (input_ids_pt, attention_mask_pt))
        traced_model.eval()

        # Define input/output specs for Core ML conversion
        states = None
        outputs = [ct.TensorType(dtype=np.float16, name="logits")]

        if use_fixed_size:
            inputs = [
                ct.TensorType(shape=(1, max_context_size), dtype=np.int32, name="inputIds"),
                ct.TensorType(shape=(1, max_context_size), dtype=np.int32, name="attentionMask"),
            ]
            outputs = [ct.TensorType(dtype=np.float16, name="logits")]
        elif use_state is False:

            # Define dynamic input dimensions with RangeDim
            seq_len_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=prompt_length)

            inputs = [
                ct.TensorType(shape=(1, seq_len_dim), dtype=np.int32, name="inputIds"),
                ct.TensorType(shape=(1, seq_len_dim), dtype=np.int32, name="attentionMask"),
            ]
        else:
            query_length = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
            end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
            inputs: List[ct.TensorType] = [
                ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
                ct.TensorType(
                    shape=(1, 1, query_length, end_step_dim),
                    dtype=np.float16,
                    name="causalMask",
                ),
            ]
            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
                    name="keyCache",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=torch_model.kv_cache_shape, dtype=np.float16),
                    name="valueCache",
                ),
            ]

        # mlmodel_f32 = ct.convert(
        #     traced_model,
        #     inputs=inputs,
        #     outputs=outputs,
        #     states=states,
        #     compute_precision=ct.precision.FLOAT32,
        #     minimum_deployment_target=ct.target.iOS18,
        # )
        # mlmodel_f32.save(f"models/{BASE_NAME}_Fp32.mlpackage")

        # Convert to Core ML model
        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            states=states,
            minimum_deployment_target=ct.target.iOS18,
        )
        mlmodel._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_ID})
        # Save the model
        mlmodel.save(f"models/{BASE_NAME}.mlpackage")
        print("Model converted and saved.")

    test_model_output(mlmodel, input_ids_pt, attention_mask_pt, use_fixed_size, use_state)
    # if os.path.exists(f"models/{BASE_NAME}_A16W8.mlpackage") is False:
    #     test_model_output(convert_model_to_a16w8(mlmodel), input_ids_pt, attention_mask_pt, use_fixed_size, use_state)
    # if os.path.exists(f"models/{BASE_NAME}_A8W8.mlpackage") is False:
    #     test_model_output(convert_model_a8w8(mlmodel), input_ids_pt, attention_mask_pt, use_fixed_size, use_state)
    # if os.path.exists(f"models/{BASE_NAME}_A8W4.mlpackage") is False:
    #     test_model_output(convert_model_a8w4(mlmodel), input_ids_pt, attention_mask_pt, use_fixed_size, use_state)
    if os.path.exists(f"models/{BASE_NAME}_Int4.mlpackage") is False:
        test_model_output(convert_model_to_int4(mlmodel), input_ids_pt, attention_mask_pt, use_fixed_size, use_state)

    


if __name__ == "__main__":
    export(use_state=True)
