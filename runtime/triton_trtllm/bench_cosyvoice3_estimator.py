# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the CosyVoice3 DiT flow-matching estimator per backend.

Data: first N samples of the seed-tts zh testset. The target speech tokens are
extracted from the ground-truth target wavs with the model's own
speech_tokenizer_v3, so the generated audio can be ASR-verified against the
target text.

Backends:
    torch       repo default (fp32, autocast off)
    torch-fp16  fp32 weights + cuda autocast fp16 (like the TRT path's autocast ONNX)
    trt         --enable-trt (autocast_fp16 ONNX, STRONGLY_TYPED engine)
    flashinfer  flashinfer-accelerated estimator (token2wav_cosyvoice3_flashinfer.py)

Usage:
    python3 bench_cosyvoice3_estimator.py --backend torch|torch-fp16|trt|flashinfer \
        [--cuda-graph] [--epochs 3] [--limit 26] --output-dir wavs_xxx
"""
import argparse
import os
import time

import torch
import torchaudio
import s3tokenizer

from token2wav_cosyvoice3 import CosyVoice3_Token2Wav

SEEDTTS_ZH = "/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/seedtts_testset/zh"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True,
                        choices=["torch", "torch-fp16", "trt", "flashinfer"])
    parser.add_argument("--model-dir", type=str, default="./Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3, help="last epoch is measured")
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cuda-graph-buckets", type=str, default=None,
                        help="comma-separated audio seconds; bucketed graphs instead of per-shape")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="samples per batch (flashinfer backend only; packed varlen)")
    return parser.parse_args()


def load_wav_16k(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav.squeeze(0)


def main():
    args = get_args()
    device = "cuda:0"
    model = CosyVoice3_Token2Wav(args.model_dir, enable_trt=(args.backend == "trt"))

    if args.backend == "torch-fp16":
        model.fp16 = True  # enables the autocast context in forward_flow
    elif args.backend == "flashinfer":
        from token2wav_cosyvoice3_flashinfer import apply_flashinfer
        buckets = ([float(s) for s in args.cuda_graph_buckets.split(",")]
                   if args.cuda_graph_buckets else None)
        apply_flashinfer(model, enable_cuda_graph=args.cuda_graph or buckets is not None,
                         cuda_graph_buckets=buckets)

    # cuda-synced timing of every estimator invocation (10 ODE steps / sample)
    decoder = model.flow.decoder
    orig_fe = decoder.forward_estimator
    stats = {"seconds": 0.0, "calls": 0}

    def timed_fe(*a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_fe(*a, **k)
        torch.cuda.synchronize()
        stats["seconds"] += time.perf_counter() - t0
        stats["calls"] += 1
        return out

    decoder.forward_estimator = timed_fe

    # ---------------- data ----------------
    metas = []
    with open(f"{SEEDTTS_ZH}/meta.lst") as f:
        for line in f:
            utt, prompt_text, prompt_wav, target_text = line.strip().split("|")
            metas.append((utt, prompt_text, prompt_wav, target_text))
            if len(metas) >= args.limit:
                break

    prompt_wavs = [load_wav_16k(os.path.join(SEEDTTS_ZH, m[2])) for m in metas]
    target_wavs = [load_wav_16k(os.path.join(SEEDTTS_ZH, "wavs", m[0] + ".wav")) for m in metas]

    # target tokens from the ground-truth wavs (untimed)
    target_tokens = []
    with torch.inference_mode():
        for wav in target_wavs:
            mels, mel_lens = s3tokenizer.padding([s3tokenizer.log_mel_spectrogram(wav)])
            codes, code_lens = model.audio_tokenizer.quantize(
                mels.to(device), mel_lens.to(device))
            target_tokens.append(codes[0, :code_lens[0].item()].tolist())

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "refs.tsv"), "w") as f:
            for utt, _, _, target_text in metas:
                f.write(f"{utt}\t{target_text}\n")

    # ---------------- benchmark ----------------
    if args.batch_size > 1:
        assert args.backend == "flashinfer", "batch>1 requires the flashinfer packed path"
        from token2wav_cosyvoice3_flashinfer import token2wav_forward_batched

    for epoch in range(args.epochs):
        stats["seconds"], stats["calls"] = 0.0, 0
        torch.manual_seed(0)
        start = time.time()
        wavs_out = []
        if args.batch_size > 1:
            for i in range(0, len(metas), args.batch_size):
                j = min(i + args.batch_size, len(metas))
                wavs = token2wav_forward_batched(
                    model, target_tokens[i:j], prompt_wavs[i:j], [16000] * (j - i))
                wavs_out.extend(wavs)
        else:
            for i in range(len(metas)):
                wavs = model([target_tokens[i]], [prompt_wavs[i]], [16000])
                wavs_out.append(wavs[0])
        torch.cuda.synchronize()
        e2e = time.time() - start
        audio_s = sum(w.shape[-1] for w in wavs_out) / 24000
        print(f"epoch {epoch}: e2e={e2e:.3f}s estimator={stats['seconds']:.3f}s "
              f"({stats['calls']} calls, {stats['seconds'] / max(stats['calls'], 1) * 1000:.2f} ms/call) "
              f"audio={audio_s:.1f}s RTF={e2e / audio_s:.4f}")

    if args.output_dir:
        for (utt, *_), wav in zip(metas, wavs_out):
            torchaudio.save(os.path.join(args.output_dir, f"{utt}.wav"),
                            wav.cpu().float(), 24000)

    tag = args.backend + ("+cudagraph" if args.cuda_graph else "") + \
        (f"+buckets[{args.cuda_graph_buckets}]" if args.cuda_graph_buckets else "") + \
        f"+b{args.batch_size}"
    print(f"RESULT backend={tag} e2e={e2e:.3f}s estimator={stats['seconds']:.3f}s "
          f"ms_per_call={stats['seconds'] / max(stats['calls'], 1) * 1000:.2f}")


if __name__ == "__main__":
    main()
