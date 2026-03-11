hf_model_path="/workspace_yuekai/HF/Fun-CosyVoice3-0.5B-2512"

onnx_path="${hf_model_path}/flow.decoder.estimator.fp32.onnx"
output_path="${hf_model_path}/flow.decoder.estimator.autocast_fp16.onnx"
calibration_data="/workspace_yuekai/tts/CosyVoice/runtime/triton_trtllm/cosyvoice3_trt_fp16_fixed/v2_autocast/cus_pg_inputs.json"
low_precision_type=fp16
log_level=INFO
data_max=65504

python3 -m modelopt.onnx.autocast \
    --onnx_path ${onnx_path} \
    --output_path ${output_path} \
    --low_precision_type ${low_precision_type} \
    --calibration_data ${calibration_data} \
    --log_level ${log_level} \
    --data_max ${data_max} \
