from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
from cosyvoice.cli.cosyvoice import AutoModel
from argparse import ArgumentParser
import numpy as np
import s3tokenizer
import soundfile as sf

def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--token2wav-path",
        type=str,
        default='/weights/Fun-CosyVoice3-0.5B-2512',
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="The prompt text",
    )
    parser.add_argument(
        "--prompt-speech-path",
        type=str,
        default="/voice_file/prompt_audio.wav",
        help="The path to the prompt speech",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default='身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。',
        help="The input text",
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
    
    args = parser.parse_args()
    return args


def audio_decode_cosyvoice(audio_tokens, tts_text, prompt_text, prompt_speech_path, codec_decoder):

    # 函数的参数要与当前版本的cosy代码对上
    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot(
        tts_text, prompt_text, prompt_speech_path, 24000, ''
    )
    tts_mel, _ = codec_decoder.model.flow.inference(
        token=audio_tokens.to(codec_decoder.model.device),
        token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(codec_decoder.model.device),
        prompt_token=model_inputs_dict["flow_prompt_speech_token"].to( codec_decoder.model.device),
        prompt_token_len=model_inputs_dict["flow_prompt_speech_token_len"].to(codec_decoder.model.device),
        prompt_feat=model_inputs_dict["prompt_speech_feat"].to(codec_decoder.model.device),
        prompt_feat_len=model_inputs_dict["prompt_speech_feat_len"].to(codec_decoder.model.device),
        embedding=model_inputs_dict["flow_embedding"].to(codec_decoder.model.device),
        finalize=True,
        streaming=True,
    )

    # 这里v2与v3不一样了
    audio_hat, _ = codec_decoder.model.hift.inference(
        # speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
        speech_feat=tts_mel, finalize=True
    )

    return audio_hat


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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval() 
    model.to(device)

    token2wav_model = AutoModel(
        model_dir=args.token2wav_path, load_trt=False, fp16=False
    )

    audio_tokenizer = s3tokenizer.load_model(args.speech_tokenizer_model_path).to(device)

    # audio_tokenizer
    waveform, sr = sf.read(args.prompt_speech_path)

    mels = []
    wav_array = torch.from_numpy(np.array(waveform, dtype=np.float32)).to(device)
    wav_len = len(wav_array)
    wav = wav_array.squeeze(0)
    mels.append(s3tokenizer.log_mel_spectrogram(wav))

    mels, mels_lens = s3tokenizer.padding(mels)
    codes, codes_lens = audio_tokenizer.quantize(mels.to(device), mels_lens.to(device))
    codes = codes.clone()
    prompt_speech_tokens = codes[0, :codes_lens[0].item()]
    prompt_speech_tokens = prompt_speech_tokens.cpu().numpy().tolist()
    prompt_speech_str = ''.join([f'<|s_{t}|>' for t in prompt_speech_tokens])

    with torch.no_grad():
        # # Tokenize the text
        chat = [
            {"role": "user", "content": f"{args.prompt_text + args.input_text}"},
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

        # Generate the speech autoregressively
        outputs = model.generate(
            input_ids,
            max_length=2048,  # We trained our model with a max length of 2048
            do_sample=True,    # True False
            top_p=0.95,           #  Adjusts the diversity of generated content
            temperature=0.8,   #  Controls randomness in output,
            repetition_penalty=1.1,
            top_k=15,           # 设置25容易多出一段重复的话
        )
        # Extract the speech tokens
        generated_ids = outputs[0][input_ids.shape[1]:-1]
        print(generated_ids.shape)

        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  

        speech_tokens = extract_speech_ids(speech_tokens)

        speech_tokens = (torch.tensor(speech_tokens)).cuda().unsqueeze(0)

        audio_hat = audio_decode_cosyvoice(
            speech_tokens,
            args.input_text,
            args.prompt_text,
            args.prompt_speech_path,
            token2wav_model,
        )

        audio = audio_hat.squeeze(0).cpu().numpy()
        sf.write("gen_streaming.wav", audio, 24000)