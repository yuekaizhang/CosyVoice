import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torch
import soundfile as sf

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
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # Split by whitespace and convert each value to int
            int_list = [int(num) for num in content.split()]
        return int_list
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
    
    # Set up the prompt speech features and speaker embedding

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


if __name__ == "__main__":

    audio_tokens = load_int_list_from_file('/workspace/slam/icefall_omni/egs/speech_llm/SPEECH2SPEECH/test_1.txt')
    # audio_tokens = [token for token in audio_tokens if token < 4096]
    audio_tokens = torch.tensor(audio_tokens, dtype=torch.int32).unsqueeze(0)
    cosyvoice = CosyVoice('/workspace/CosyVoice-300M-SFT', load_jit=False, load_trt=False, fp16=False)
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    audio_hat = audio_decode_cosyvoice(audio_tokens, cosyvoice)
    sf.write('output_22050_new.wav', audio_hat.squeeze(0).cpu().numpy(), 22050)