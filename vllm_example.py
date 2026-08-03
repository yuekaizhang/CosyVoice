import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm


def cosyvoice2_example():
    """ CosyVoice2 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_example():
    """ CosyVoice3 vllm usage
    """
    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, load_vllm=True, fp16=False)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for _, _ in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                            './asset/zero_shot_prompt.wav', stream=False)):
            continue


def cosyvoice3_spec_example():
    """CosyVoice3 + DSpark speculative decoding.

    Requires:
    1. A HuggingFace-format export of the CosyVoice3 LLM (produced by
       runtime/triton_trtllm/scripts/convert_cosyvoice3_to_hf.py).
    2. A speculators-format DSpark drafter checkpoint (local path or HF repo).
    3. The speculative vllm fork (yuekaizhang/vllm, branch
       dspark-draft-sampling-mirrors) loaded via PYTHONPATH with compiled
       extensions from a vllm-omni 0.25.1 build.

    Example launch (set PYTHONPATH before starting Python):
        PYTHONPATH=/path/to/speculative/vllm:/path/to/vllm025_venv/lib/python3.12/site-packages \\
            python vllm_example.py
    """
    DRAFT_MODEL = 'yuekai/cosyvoice3_llm_dspark'  # or local path to checkpoint_best/

    cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_trt=True, fp16=False)
    # HF export is created automatically under pretrained_models/Fun-CosyVoice3-0.5B/hf_spec
    cosyvoice.model.load_vllm_spec('pretrained_models/Fun-CosyVoice3-0.5B', DRAFT_MODEL)

    for i in tqdm(range(10)):
        set_all_random_seed(i)
        for j, result in enumerate(cosyvoice.inference_zero_shot(
                '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
                'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                './asset/zero_shot_prompt.wav', stream=False)):
            result['tts_speech'].numpy().tofile(f'spec_{i}_{j}.pcm')


def main():
    # cosyvoice2_example()
    # cosyvoice3_example()
    cosyvoice3_spec_example()


if __name__ == '__main__':
    main()
