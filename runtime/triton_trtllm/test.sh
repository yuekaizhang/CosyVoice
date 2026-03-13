#!/bin/bash
# Copyright (c) 2026 NVIDIA (authors: Yuekai Zhang)
export CUDA_VISIBLE_DEVICES=0
cosyvoice_path=/workspace_yuekai/tts/CosyVoice

export PYTHONPATH=${cosyvoice_path}:$PYTHONPATH
export PYTHONPATH=${cosyvoice_path}/third_party/Matcha-TTS:$PYTHONPATH

stage=$1
stop_stage=$2

huggingface_model_local_dir=./hf_cosyvoice3_llm
model_scope_model_local_dir=/workspace_yuekai/HF/Fun-CosyVoice3-0.5B-2512

trt_dtype=bfloat16
trt_weights_dir=./trt_weights_${trt_dtype}
trt_engines_dir=./trt_engines_${trt_dtype}

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    python3 test_cosy3_token2wav.py \
     --model-dir $model_scope_model_local_dir \
     --llm-path $huggingface_model_local_dir \
     --prompt-speech-path ./prompt_audio.wav \
     --streaming --enable-trt \
     --output-path test_token2wav_output_streaming_rerun_trt_fix.wav
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    python3 ${cosyvoice_path}/cosyvoice/bin/export_onnx_streaming.py \
     --model_dir $model_scope_model_local_dir
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    python3 infer_cosy3_mairhub.py \
     --token2wav-path $model_scope_model_local_dir \
     --prompt-speech-path ./prompt_audio.wav \
     --model-path $huggingface_model_local_dir \
     --speech_tokenizer_model_path $model_scope_model_local_dir/speech_tokenizer_v3.onnx
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
    python3 infer_cosy3_mairhub.py \
     --token2wav-path $model_scope_model_local_dir \
     --prompt-speech-path ./prompt_audio.wav \
     --model-path $huggingface_model_local_dir \
     --speech_tokenizer_model_path $model_scope_model_local_dir/speech_tokenizer_v3.onnx \
     --streaming
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
    python3 infer_cosy3_mairhub_dataset.py \
     --token2wav-path $model_scope_model_local_dir \
     --model-path $huggingface_model_local_dir \
     --speech_tokenizer_model_path $model_scope_model_local_dir/speech_tokenizer_v3.onnx \
     --huggingface-dataset-split wenetspeech4tts --streaming \
     --output-dir ./streaming_dataset_output_stage13
fi

if [ $stage -le 15 ] && [ $stop_stage -ge 15 ]; then
    echo "===== Stage 15: Compare prompt features and streaming audio ====="
    python3 debug_compare_features.py \
     --model-dir $model_scope_model_local_dir \
     --llm-path $huggingface_model_local_dir \
     --prompt-speech-path ./prompt_audio.wav
fi