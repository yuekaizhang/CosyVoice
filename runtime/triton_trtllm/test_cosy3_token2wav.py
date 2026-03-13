""" Test CosyVoice3 Token2Wav with a single prompt audio.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 test_cosy3_token2wav.py
"""
import torch
import torchaudio
import numpy as np
import s3tokenizer
import soundfile as sf
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from token2wav_cosyvoice3 import CosyVoice3_Token2Wav


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=str,
                        default="/workspace_yuekai/HF/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--llm-path", type=str,
                        default="./hf_cosyvoice3_llm")
    parser.add_argument("--prompt-speech-path", type=str,
                        default="./prompt_audio.wav")
    parser.add_argument("--prompt-text", type=str,
                        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。")
    parser.add_argument("--input-text", type=str,
                        default="身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。")
    parser.add_argument("--enable-trt", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--output-path", type=str, default="test_token2wav_output.wav")
    return parser.parse_args()


def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            speech_ids.append(int(token_str[4:-2]))
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda")

    # 1. Load LLM tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_path)
    llm_model.eval().to(device)

    # 2. Load Token2Wav model (default: no TRT)
    token2wav = CosyVoice3_Token2Wav(
        model_dir=args.model_dir, enable_trt=args.enable_trt,
        streaming=args.streaming,
    )

    # 3. Read and prepare prompt audio (resample to 16kHz if needed)
    waveform, sr = sf.read(args.prompt_speech_path)
    prompt_audio = torch.from_numpy(np.array(waveform, dtype=np.float32))
    if prompt_audio.dim() > 1:
        prompt_audio = prompt_audio[:, 0]
    if sr != 16000:
        prompt_audio = torchaudio.transforms.Resample(sr, 16000)(prompt_audio.unsqueeze(0)).squeeze(0)
        sr = 16000

    # 4. Tokenize prompt audio for LLM input
    audio_tokenizer = s3tokenizer.load_model(
        f"{args.model_dir}/speech_tokenizer_v3.onnx"
    ).to(device)
    mels = [s3tokenizer.log_mel_spectrogram(prompt_audio.to(device))]
    mels, mels_lens = s3tokenizer.padding(mels)
    codes, codes_lens = audio_tokenizer.quantize(mels.to(device), mels_lens.to(device))
    prompt_speech_tokens = codes[0, :codes_lens[0].item()].cpu().numpy().tolist()
    prompt_speech_str = ''.join([f'<|s_{t}|>' for t in prompt_speech_tokens])

    # 5. LLM generate speech tokens
    with torch.no_grad():
        chat = [
            {"role": "user", "content": 'You are a helpful assistant.<|endofprompt|>' + args.prompt_text + args.input_text},
            {"role": "assistant", "content": prompt_speech_str}
        ]
        input_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors='pt', continue_final_message=True
        ).to(device)

        outputs = llm_model.generate(
            input_ids, max_length=2048,
            do_sample=True, top_p=0.95, temperature=0.8,
            repetition_penalty=1.1, top_k=15,
        )
        generated_ids = outputs[0][input_ids.shape[1]:-1]
        speech_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        speech_ids = extract_speech_ids(speech_tokens_str)
        print(f"Generated {len(speech_ids)} speech tokens")

    # 6. Token2Wav: convert speech tokens to audio
    generated_wavs = token2wav(
        generated_speech_tokens_list=[speech_ids],
        prompt_audios_list=[prompt_audio],
        prompt_audios_sample_rate=[16000],
        streaming=args.streaming,
    )

    # 7. Save output
    torchaudio.save(args.output_path, generated_wavs[0].cpu(), 24000)
    print(f"Saved to {args.output_path}")
