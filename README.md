# GPTQ-for-KoAlpaca
[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)를 사용하여 [GPTQ 양자화 알고리즘](https://arxiv.org/abs/2210.17323)을 [KoAlpaca](https://github.com/Beomi/KoAlpaca)에 적용하기 위한 코드입니다.
## 설치
```
conda create -n koalpaca python=3.9
conda activate koalpaca
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install auto-gptq
```
## 양자화
```
python quant_with_alpaca.py --pretrained_model_dir beomi/KoAlpaca-Polyglot-5.8B --quantized_model_dir ./model
```
양자화 후 model 파일에 GPTQ 양자화된 가중치가 저장됩니다.

아래 예제 코드로 실행해볼 수 있습니다.
``` python
import torch
from transformers import pipeline
from auto_gptq import AutoGPTQForCausalLM
MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'
QUANT_MODEL = './model'

model = AutoGPTQForCausalLM.from_quantized(QUANT_MODEL, device="cuda:0", use_triton=False)

pipe = pipeline('text-generation', model=model,tokenizer=MODEL,device=0)

def ask(x, context='', is_input_full=False):
    ans = pipe(
        f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {x}\n\n### 답변:", 
        do_sample=True, 
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    print(ans[0]['generated_text'])

ask("딥러닝이 뭐야?")
# 딥러닝은 인공 신경망을 통해 더 복잡한 문제를 해결하는 기술입니다. 머신러닝과 달리, 사람이 수행하는 작업을 수행하는 데 큰 도움을 줄 수 있습니다. 예를 들어, 이미지 인식과 같은 분야에서 뛰어난 성능을 보여주고 있습니다. 최근에는 딥러닝을 활용한 인공지능 서비스도 등장하면서 주목받고 있습니다.  더 자세한 설명: 딥러닝은 머신러닝과는 달리, 사람이 수행하는 작업을 보다 복잡한 알고리즘을 통해 수행합니다. 인공 신경망을 구성하는 것부터 시작하여, 데이터를 처리하고 인식하는 작업 등 모든 계산과 복잡한 과정을 거쳐서 최종적인 결과를 도출해 냅니다. 이러한 이유로 딥러닝은 머신러닝보다 더 높은 성능을 보여주면서도, 더 복잡한 알고리즘에 기초하여 문제를 해결할 수 있습니다.
```
