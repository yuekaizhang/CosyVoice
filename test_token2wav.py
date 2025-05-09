import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torch
import soundfile as sf
from lhotse import load_manifest_lazy
from datasets import load_dataset

def load_tokens_from_lhotse_manifest(manifest_path):
    """
    Load tokens from a Lhotse manifest file.

    Args:
        manifest_path (str): Path to the Lhotse manifest file
        
    Returns:
        list: A list of tokens read from the manifest file
        
    Raises:
        FileNotFoundError: If the manifest file doesn't exist
        ValueError: If the manifest file contains non-integer values
    """
    manifest = load_manifest_lazy(manifest_path)
    tokens = []
    for cut in manifest:
        tokens.append((cut.id, cut.custom['speech_token']))
    return tokens

def load_int_list_from_file(file_path):
    """
    Load space-separated integers from a file into a list of integers.
    
    Args:
        file_path (str): Path to the file to be loaded
        
    Returns:
        list: A list of integers read from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains non-integer values
    """
    results = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                wav_name, content = line.split('|')
                int_list = [int(num) for num in content.split()]
                results.append((wav_name, int_list))
        return results
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError as e:
        raise ValueError(f"File contains non-integer values: {str(e)}")

def audio_decode_cosyvoice(audio_tokens, codec_decoder):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        tone_dir (str): The tone directory or setting.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Generated audio waveform.
    """
    
    flow_embedding = codec_decoder.frontend.spk2info['中文女']['embedding']
    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
    prompt_speech_feat = torch.zeros(1, 0, 80)

    tts_mel, _ = codec_decoder.model.flow.inference(token=audio_tokens.to(codec_decoder.model.device),
                                                token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
                                                prompt_token=flow_prompt_speech_token.to(codec_decoder.model.device),
                                                prompt_token_len=torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
                                                prompt_feat=prompt_speech_feat.to(codec_decoder.model.device),
                                                prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
                                                embedding=flow_embedding.to(codec_decoder.model.device),
                                                flow_cache=torch.zeros(1, 80, 0, 2).to(codec_decoder.model.device),)


    audio_hat, _ = codec_decoder.model.hift.inference(speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0))

    return audio_hat

def audio_decode_cosyvoice2(audio_tokens, prompt_text, prompt_speech_16k, codec_decoder):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        tone_dir (str): The tone directory or setting.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Generated audio waveform.
    """
    # model_inputs_dict = codec_decoder.frontend.spk2info['英文女']
    # flow_embedding = model_inputs_dict['embedding']
    # flow_prompt_speech_token = model_inputs_dict['speech_token']
    # prompt_speech_feat = model_inputs_dict['speech_feat']

    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot('empty', prompt_text, prompt_speech_16k, 24000)
    tts_mel, _ = codec_decoder.model.flow.inference(token=audio_tokens.to(codec_decoder.model.device),
                                                token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
                                                prompt_token=model_inputs_dict['flow_prompt_speech_token'].to(codec_decoder.model.device),
                                                prompt_token_len=torch.tensor([model_inputs_dict['flow_prompt_speech_token_len']], dtype=torch.int32).to(codec_decoder.model.device),
                                                prompt_feat=model_inputs_dict['prompt_speech_feat'].to(codec_decoder.model.device),
                                                prompt_feat_len=model_inputs_dict['prompt_speech_feat_len'].to(codec_decoder.model.device),
                                                embedding=model_inputs_dict['flow_embedding'].to(codec_decoder.model.device), finalize=True,
                                                # cache=torch.zeros(1, 80, 0, 2).to(codec_decoder.model.device),
                                                )
    # tts_mel, _ = codec_decoder.model.flow.inference(token=audio_tokens.to(codec_decoder.model.device),
    #                                             token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
    #                                             prompt_token=flow_prompt_speech_token.to(codec_decoder.model.device),
    #                                             prompt_token_len=torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
    #                                             prompt_feat=prompt_speech_feat.to(codec_decoder.model.device),
    #                                             prompt_feat_len=torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
    #                                             embedding=flow_embedding.to(codec_decoder.model.device), finalize=True,
    #                                             # cache=torch.zeros(1, 80, 0, 2).to(codec_decoder.model.device),
    #                                             )


    audio_hat, _ = codec_decoder.model.hift.inference(speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0))

    return audio_hat

if __name__ == "__main__":
    # items = load_int_list_from_file('/workspace/slam/icefall_omni/egs/speech_llm/SPEECH2SPEECH/tokens2.txt')
    items = load_tokens_from_lhotse_manifest('/workspace/slam/icefall_omni/egs/speech_llm/SPEECH2SPEECH/data/fbank/cuts_debug.jsonl.gz')
    # seed_data = load_dataset("yuekai/seed_tts_cosy2", split='wenetspeech4tts')

    # cosyvoice = CosyVoice('/workspace/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
    cosyvoice = CosyVoice2('/workspace/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
    # for i, item in enumerate(seed_data):
    #     if i > 2:
    #         break
    #     prompt_text = item['prompt_text']
    #     prompt_speech_16k = item['prompt_audio']['array']
    #     prompt_speech_16k = torch.from_numpy(prompt_speech_16k).unsqueeze(0).to(torch.float32)
    #     prompt_speech_16k = prompt_speech_16k.mean(dim=0, keepdim=True)
    #     # print(prompt_speech_16k.shape, type(prompt_speech_16k), prompt_speech_16k.dtype)
    #     # exit(0)
    #     # print(prompt_speech_16k.shape, type(prompt_speech_16k))
    #     # exit(0)
    #     audio_tokens = item['target_audio_cosy2_tokens']
    #     id = item['id']
    #     audio_hat = audio_decode_cosyvoice2(audio_tokens, prompt_text, prompt_speech_16k, cosyvoice)
    #     sf.write(f'{item["wav_name"]}.wav', audio_hat.squeeze(0).cpu().numpy(), 24000)



    prompt_text = 'Some call me nature, others call me mother nature.'
    #prompt_text = 'Romeo and Juliet might be the most famous act of William Shakespeare.'
    prompt_speech_16k = load_wav('./basic_en.wav', 16000)
    #prompt_speech_16k = load_wav('common_voice_en_2586258.wav', 16000)

    for wav_name, audio_tokens in items:
        print(f'Processing {wav_name}...')
        # audio_tokens = [token for token in audio_tokens if token < 4096]
        audio_tokens = torch.tensor(audio_tokens, dtype=torch.int32).unsqueeze(0)

        audio_hat = audio_decode_cosyvoice2(audio_tokens, prompt_text, prompt_speech_16k, cosyvoice)
        # sf.write(f'{wav_name}.wav', audio_hat.squeeze(0).cpu().numpy(), 22050)
        sf.write(f'{wav_name}.wav', audio_hat.squeeze(0).cpu().numpy(), 24000)