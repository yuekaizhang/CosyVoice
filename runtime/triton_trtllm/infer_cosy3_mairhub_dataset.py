from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torchaudio
import soundfile as sf
from cosyvoice.cli.cosyvoice import AutoModel
from argparse import ArgumentParser
from datasets import load_dataset
import numpy as np
import os
import time
import s3tokenizer


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--token2wav-path",
        type=str,
        default='/weights/Fun-CosyVoice3-0.5B-2512',
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default='/weights/transformers_cosyvoice3_llm',
        help="The path to the model",
    )
    parser.add_argument(
        "--speech_tokenizer_model_path",
        type=str,
        default='/weights/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx',
        help="path to speech_tokenizer_v3.onnx",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Enable streaming audio decode (process tokens in chunks)",
    )
    parser.add_argument(
        "--huggingface-dataset-split",
        type=str,
        default="wenetspeech4tts",
        help="HuggingFace dataset split name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_output",
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Max number of samples to process (-1 for all)",
    )

    args = parser.parse_args()
    return args


def audio_decode_cosyvoice(audio_tokens, tts_text, prompt_text, prompt_audio, prompt_sr, codec_decoder):
    """Non-streaming audio decode using prompt audio array directly."""
    # Write prompt audio to a temp file for frontend_zero_shot
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, prompt_audio, prompt_sr)

    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot(
        tts_text, prompt_text, tmp_path, 24000, ''
    )
    os.unlink(tmp_path)

    tts_mel, _ = codec_decoder.model.flow.inference(
        token=audio_tokens.to(codec_decoder.model.device),
        token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
        prompt_token=model_inputs_dict["flow_prompt_speech_token"].to(codec_decoder.model.device),
        prompt_token_len=model_inputs_dict["flow_prompt_speech_token_len"].to(codec_decoder.model.device),
        prompt_feat=model_inputs_dict["prompt_speech_feat"].to(codec_decoder.model.device),
        prompt_feat_len=model_inputs_dict["prompt_speech_feat_len"].to(codec_decoder.model.device),
        embedding=model_inputs_dict["flow_embedding"].to(codec_decoder.model.device),
        finalize=True,
        streaming=False,
    )

    audio_hat, _ = codec_decoder.model.hift.inference(
        speech_feat=tts_mel, finalize=True
    )
    return audio_hat


def audio_decode_cosyvoice_stream(audio_tokens, tts_text, prompt_text, prompt_audio, prompt_sr, codec_decoder,
                                   token_hop_len=25, stream_scale_factor=2, token_max_hop_len=100):
    """Streaming audio decode using prompt audio array directly."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, prompt_audio, prompt_sr)

    device = codec_decoder.model.device

    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot(
        tts_text, prompt_text, tmp_path, 24000, ''
    )
    os.unlink(tmp_path)

    flow_prompt_speech_token = model_inputs_dict["flow_prompt_speech_token"].to(device)
    flow_prompt_speech_token_len = model_inputs_dict["flow_prompt_speech_token_len"].to(device)
    prompt_speech_feat = model_inputs_dict["prompt_speech_feat"].to(device)
    prompt_speech_feat_len = model_inputs_dict["prompt_speech_feat_len"].to(device)
    flow_embedding = model_inputs_dict["flow_embedding"].to(device)

    pre_lookahead_len = codec_decoder.model.flow.pre_lookahead_len
    token_mel_ratio = codec_decoder.model.flow.token_mel_ratio

    prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / token_hop_len) * token_hop_len - flow_prompt_speech_token.shape[1])

    total_tokens = audio_tokens.shape[1]
    token_offset = 0
    current_hop = token_hop_len
    hift_cache_mel = None
    speech_offset = 0
    audio_chunks = []

    while token_offset < total_tokens:
        this_hop = current_hop + prompt_token_pad if token_offset == 0 else current_hop
        remaining = total_tokens - token_offset

        if remaining >= this_hop + pre_lookahead_len:
            end_idx = token_offset + this_hop + pre_lookahead_len
            this_token = audio_tokens[:, :end_idx].to(device, dtype=torch.int32)
            finalize = False
        else:
            this_token = audio_tokens.to(device, dtype=torch.int32)
            finalize = True

        tts_mel, _ = codec_decoder.model.flow.inference(
            token=this_token,
            token_len=torch.tensor([this_token.shape[1]], dtype=torch.int32).to(device),
            prompt_token=flow_prompt_speech_token,
            prompt_token_len=flow_prompt_speech_token_len,
            prompt_feat=prompt_speech_feat,
            prompt_feat_len=prompt_speech_feat_len,
            embedding=flow_embedding,
            streaming=True,
            finalize=finalize,
        )

        tts_mel = tts_mel[:, :, token_offset * token_mel_ratio:]

        if hift_cache_mel is not None:
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        hift_cache_mel = tts_mel

        tts_speech, _ = codec_decoder.model.hift.inference(speech_feat=tts_mel, finalize=finalize)
        tts_speech = tts_speech[:, speech_offset:]
        speech_offset += tts_speech.shape[1]

        audio_chunks.append(tts_speech.cpu())

        token_offset += this_hop
        if not finalize:
            current_hop = min(token_max_hop_len, current_hop * stream_scale_factor)
        else:
            break

    return torch.cat(audio_chunks, dim=1)


def extract_speech_ids(speech_tokens_str):
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()
    model.to(device)

    token2wav_model = AutoModel(
        model_dir=args.token2wav_path, load_trt=False, fp16=False
    )

    audio_tokenizer = s3tokenizer.load_model(args.speech_tokenizer_model_path).to(device)

    dataset = load_dataset("yuekai/seed_tts_cosy2", split=args.huggingface_dataset_split,
                          trust_remote_code=True)

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Processing {len(dataset)} samples, output_dir={args.output_dir}, streaming={args.streaming}")

    target_sample_rate = 16000
    total_time = 0

    for idx, item in enumerate(dataset):
        start_time = time.time()

        prompt_text = item["prompt_text"]
        target_text = item["target_text"]
        sample_id = item.get("id", f"sample_{idx:06d}")

        # Get prompt audio and resample to 16kHz for s3 tokenizer
        ref_audio = torch.from_numpy(np.array(item["prompt_audio"]["array"], dtype=np.float32))
        ref_sr = item["prompt_audio"]["sampling_rate"]
        if ref_sr != target_sample_rate:
            ref_audio_16k = torchaudio.transforms.Resample(ref_sr, target_sample_rate)(ref_audio.unsqueeze(0)).squeeze(0)
        else:
            ref_audio_16k = ref_audio

        # Extract speech tokens from prompt audio using s3 tokenizer
        mels = [s3tokenizer.log_mel_spectrogram(ref_audio_16k)]
        mels_padded, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = audio_tokenizer.quantize(mels_padded.to(device), mels_lens.to(device))
        codes = codes.clone()
        prompt_speech_tokens = codes[0, :codes_lens[0].item()]
        prompt_speech_tokens_list = prompt_speech_tokens.cpu().numpy().tolist()
        prompt_speech_str = ''.join([f'<|s_{t}|>' for t in prompt_speech_tokens_list])

        with torch.no_grad():
            chat = [
                {"role": "user", "content": f"{'You are a helpful assistant.<|endofprompt|>'+ prompt_text + target_text}"},
                {"role": "assistant", "content": prompt_speech_str}
            ]
            assert 'system' not in tokenizer.chat_template, "system should not be in chat_template"

            input_ids = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_tensors='pt',
                continue_final_message=True
            )
            input_ids = input_ids.to(device)

            outputs = model.generate(
                input_ids,
                max_length=2048,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                repetition_penalty=1.1,
                top_k=25,
            )

            generated_ids = outputs[0][input_ids.shape[1]:-1]
            speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = extract_speech_ids(speech_tokens)
            speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0)

            # Use original prompt audio (not resampled) for token2wav
            prompt_audio_np = np.array(item["prompt_audio"]["array"], dtype=np.float32)
            prompt_audio_sr = item["prompt_audio"]["sampling_rate"]

            if args.streaming:
                audio_hat = audio_decode_cosyvoice_stream(
                    speech_tokens,
                    target_text,
                    prompt_text,
                    prompt_audio_np,
                    prompt_audio_sr,
                    token2wav_model,
                )
            else:
                audio_hat = audio_decode_cosyvoice(
                    speech_tokens,
                    target_text,
                    prompt_text,
                    prompt_audio_np,
                    prompt_audio_sr,
                    token2wav_model,
                )

            audio = audio_hat.squeeze(0).cpu().numpy()
            output_path = os.path.join(args.output_dir, f"{sample_id}.wav")
            sf.write(output_path, audio, 24000)

        elapsed = time.time() - start_time
        total_time += elapsed
        print(f"[{idx+1}/{len(dataset)}] {sample_id} done, tokens={generated_ids.shape[0]}, time={elapsed:.2f}s")

    print(f"All done. Total time: {total_time:.2f}s, avg: {total_time/len(dataset):.2f}s/sample")
