import torch
from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import s3tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchaudio
import os
import logging
import argparse
import queue
import time


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")

class TrtContextWrapper:
    def __init__(self, trt_engine, trt_concurrent=1, device='cuda:0'):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            trt_stream = torch.cuda.stream(torch.cuda.Stream(device))
            assert trt_context is not None, 'failed to create trt context, maybe not enough CUDA memory, try reduce current trt concurrent {}'.format(trt_concurrent)
            self.trt_context_pool.put([trt_context, trt_stream])
        assert self.trt_context_pool.empty() is False, 'no avaialbe estimator context'

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])

class CosyVoice2_Token2Wav(torch.nn.Module):
    def __init__(self, model_dir: str = "./CosyVoice2-0.5B", enable_trt: bool = False):
        super().__init__()
        self.flow = CausalMaskedDiffWithXvec()
        self.flow.half()
        self.flow.load_state_dict(torch.load(f"{model_dir}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_dir}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_dir}/campplus.onnx", sess_options=option,
                                                    providers=["CPUExecutionProvider"])
        
        self.audio_tokenizer = s3tokenizer.load_model(f"{model_dir}/speech_tokenizer_v2.onnx").cuda().eval()
        gpu="a100"
        #gpu="h100"
        #gpu="l20"
        if enable_trt:
            self.load_trt(f'{model_dir}/flow.decoder.estimator.fp16.dynamic_batch.{gpu}.plan',
                                f'{model_dir}/flow.decoder.estimator.fp32.dynamic_batch.onnx',
                                1,
                                True)
            self.load_spk_trt(f'{model_dir}/campplus.{gpu}.fp32.trt',
                                f'{model_dir}/campplus.onnx',
                                1,
                                False)


    def forward_spk_embedding(self, spk_feat):
        if isinstance(self.spk_model, onnxruntime.InferenceSession):
            return self.spk_model.run(
                None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()
        else:
            [spk_model, stream], trt_engine = self.spk_model.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            spk_feat = spk_feat.unsqueeze(dim=0).cuda()
            batch_size = spk_feat.size(0)
            # output shape (batch_size, 192)
            with stream:
                spk_model.set_input_shape('input', (batch_size, spk_feat.size(1), 80))
                output_tensor = torch.empty((batch_size, 192), device=spk_feat.device)
                # print(output_tensor.shape, output_tensor, "output_tensor")
                data_ptrs = [spk_feat.contiguous().data_ptr(),
                             output_tensor.contiguous().data_ptr()]
                for i, j in enumerate(data_ptrs):
                    # print(trt_engine.get_tensor_name(i))
                    spk_model.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert spk_model.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.spk_model.release_estimator(spk_model, stream)
            # print(output_tensor.shape, output_tensor)
            # input()
            # input()
            return output_tensor.cpu().numpy().flatten().tolist()

    def load_spk_trt(self, spk_model, spk_onnx_model, trt_concurrent=1, fp16=True):
        if not os.path.exists(spk_model) or os.path.getsize(spk_model) == 0:
            trt_kwargs = self.get_spk_trt_kwargs()
            convert_onnx_to_trt(spk_model, trt_kwargs, spk_onnx_model, fp16)
        import tensorrt as trt
        with open(spk_model, 'rb') as f:
            spk_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert spk_engine is not None, 'failed to load trt {}'.format(spk_model)
        self.spk_model = TrtContextWrapper(spk_engine, trt_concurrent=trt_concurrent)

    def get_spk_trt_kwargs(self):
        min_shape = [(1, 4, 80)]
        opt_shape = [(1, 500, 80)]
        max_shape = [(1, 3000, 80)]
        input_names = ["input"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent=1, fp16=True):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            trt_kwargs = self.get_trt_kwargs_dynamic_batch(opt_batch_size=2, max_batch_size=16)
            convert_onnx_to_trt(flow_decoder_estimator_model, trt_kwargs, flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def get_trt_kwargs_dynamic_batch(self, opt_batch_size=2, max_batch_size=64):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4), (2,), (2, 80)]
        opt_shape = [(opt_batch_size*2, 80, 500), (opt_batch_size*2, 1, 500), (opt_batch_size*2, 80, 500), (opt_batch_size*2, 80, 500), (opt_batch_size*2,), (opt_batch_size*2, 80)]
        max_shape = [(max_batch_size*2, 80, 3000), (max_batch_size*2, 1, 3000), (max_batch_size*2, 80, 3000), (max_batch_size*2, 80, 3000), (max_batch_size*2,), (max_batch_size*2, 80)]
        input_names = ["x", "mask", "mu", "cond", "t", "spks"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def prompt_audio_tokenization(self, prompt_audios_list: list[torch.Tensor]) -> list[list[int]]:
        prompt_speech_tokens_list, prompt_speech_mels_list = [], []
        for audio in prompt_audios_list:
            assert len(audio.shape) == 1
            log_mel = s3tokenizer.log_mel_spectrogram(audio)  # [num_mels, T]
            prompt_speech_mels_list.append(log_mel)
        prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(prompt_speech_mels_list)
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            prompt_mels_for_llm.cuda(), prompt_mels_lens_for_llm.cuda()
        )
        for i in range(len(prompt_speech_tokens)):
            speech_tokens_i = prompt_speech_tokens[i, :prompt_speech_tokens_lens[i].item()].tolist()
            prompt_speech_tokens_list.append(speech_tokens_i)
        return prompt_speech_tokens_list
    
    def get_spk_emb(self, prompt_audios_list: list[torch.Tensor]) -> torch.Tensor:
        spk_emb_for_flow = []
        for audio in prompt_audios_list:
            assert len(audio.shape) == 1
            spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
            # spk_emb = self.spk_model.run(
            #     None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            # )[0].flatten().tolist()
            spk_emb = self.forward_spk_embedding(spk_feat)

            spk_emb_for_flow.append(spk_emb)
        spk_emb_for_flow = torch.tensor(spk_emb_for_flow)    
        return spk_emb_for_flow
    
    def get_prompt_mels(self, prompt_audios_list: list[torch.Tensor], prompt_audios_sample_rate: list[int]):
        prompt_mels_for_flow = []
        prompt_mels_lens_for_flow = []
        for audio, sample_rate in zip(prompt_audios_list, prompt_audios_sample_rate):
            assert len(audio.shape) == 1
            audio = audio.unsqueeze(0)
            if sample_rate != 24000:
                audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
            mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
            mel_len = mel.shape[0]
            prompt_mels_for_flow.append(mel)
            prompt_mels_lens_for_flow.append(mel_len)
        prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(prompt_mels_for_flow, batch_first=True, padding_value=0)  # [B, T', num_mels=80]
        prompt_mels_lens_for_flow = torch.tensor(prompt_mels_lens_for_flow)
        return prompt_mels_for_flow, prompt_mels_lens_for_flow
    

    def forward_flow(self, prompt_speech_tokens_list: list[list[int]], generated_speech_tokens_list: list[list[int]], prompt_mels_for_flow: torch.Tensor, prompt_mels_lens_for_flow: torch.Tensor, spk_emb_for_flow: torch.Tensor):
        batch_size = prompt_mels_for_flow.shape[0]
        flow_inputs = []
        flow_inputs_lens = []
        for prompt_speech_tokens, generated_speech_tokens in zip(prompt_speech_tokens_list, generated_speech_tokens_list):
            flow_inputs.append(torch.tensor(prompt_speech_tokens + generated_speech_tokens))
            flow_inputs_lens.append(len(prompt_speech_tokens) + len(generated_speech_tokens))

        flow_inputs = torch.nn.utils.rnn.pad_sequence(flow_inputs, batch_first=True, padding_value=0)
        flow_inputs_lens = torch.tensor(flow_inputs_lens)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            generated_mels, generated_mels_lens = self.flow(
                flow_inputs.cuda(), flow_inputs_lens.cuda(),
                prompt_mels_for_flow.cuda(), prompt_mels_lens_for_flow.cuda(), spk_emb_for_flow.cuda(),
                streaming=False, finalize=True
            )

        return generated_mels, generated_mels_lens

    def forward_hift(self, generated_mels: torch.Tensor, generated_mels_lens: torch.Tensor, prompt_mels_lens_for_flow: torch.Tensor):
        batch_size = generated_mels.shape[0]
        generated_wavs = []
        for i in range(batch_size):
            mel = generated_mels[i, :, prompt_mels_lens_for_flow[i].item():generated_mels_lens[i].item()].unsqueeze(0)
            wav, _ = self.hift(speech_feat=mel)
            generated_wavs.append(wav)
        return generated_wavs


    @torch.inference_mode()
    def forward(
        self, generated_speech_tokens_list: list[list[int]], prompt_audios_list: list[torch.Tensor], prompt_audios_sample_rate: list[int]
    ):
        # assert all item in prompt_audios_sample_rate is 16000
        assert all(sample_rate == 16000 for sample_rate in prompt_audios_sample_rate)
        
        # Create CUDA events for precise timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Synchronize before starting
        torch.cuda.synchronize()
        start_event.record()
        prompt_speech_tokens_list = self.prompt_audio_tokenization(prompt_audios_list)
        end_event.record()
        torch.cuda.synchronize()
        print(f"prompt_audio_tokenization taken: {start_event.elapsed_time(end_event):.4f} ms")
        
        start_event.record()
        prompt_mels_for_flow, prompt_mels_lens_for_flow = self.get_prompt_mels(prompt_audios_list, prompt_audios_sample_rate)
        end_event.record()
        torch.cuda.synchronize()
        print(f"get_prompt_mels taken: {start_event.elapsed_time(end_event):.4f} ms")
        
        start_event.record()
        spk_emb_for_flow = self.get_spk_emb(prompt_audios_list)
        end_event.record()
        torch.cuda.synchronize()
        print(f"get_spk_emb taken: {start_event.elapsed_time(end_event):.4f} ms")
        
        start_event.record()
        generated_mels, generated_mels_lens = self.forward_flow(prompt_speech_tokens_list, generated_speech_tokens_list, prompt_mels_for_flow, prompt_mels_lens_for_flow, spk_emb_for_flow)
        end_event.record()
        torch.cuda.synchronize()
        print(f"forward_flow taken: {start_event.elapsed_time(end_event):.4f} ms")
        
        start_event.record()
        generated_wavs = self.forward_hift(generated_mels, generated_mels_lens, prompt_mels_lens_for_flow)
        end_event.record()
        torch.cuda.synchronize()
        print(f"forward_hift taken: {start_event.elapsed_time(end_event):.4f} ms")
        print(f"--------------------------------")
        return generated_wavs


def collate_fn(batch):
    ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate = [], [], [], []
    for i, item in enumerate(batch):
        generated_speech_tokens_list.append(item['target_audio_cosy2_tokens'])
        audio = torch.from_numpy(item['prompt_audio']['array']).float() 
        prompt_audios_list.append(audio)
        prompt_audios_sample_rate.append(item['prompt_audio']['sampling_rate'])
        ids.append(item['id'])

    return ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-trt", action="store_true")
    parser.add_argument("--model-dir", type=str, default="./CosyVoice2-0.5B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="generated_wavs")
    parser.add_argument("--huggingface-dataset-split", type=str, default="wenetspeech4tts")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model = CosyVoice2_Token2Wav(model_dir=args.model_dir, enable_trt=args.enable_trt)
    # mkdir output_dir if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    dataset_name = "yuekai/seed_tts_cosy2"

    dataset = load_dataset(dataset_name, split=args.huggingface_dataset_split, trust_remote_code=True)


    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    for _ in range(3):
        start_time = time.time()
        for batch in data_loader:
            ids, generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate = batch

            generated_wavs = model(generated_speech_tokens_list, prompt_audios_list, prompt_audios_sample_rate)

            for id, wav in zip(ids, generated_wavs):
                torchaudio.save(f"{args.output_dir}/{id}.wav", wav.cpu(), 24000)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

    with open(f"{args.output_dir}/log.txt", "w") as f:
        f.write(f"Time taken: {end_time - start_time} seconds\n")
        # write args to log.txt
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")