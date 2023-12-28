
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import numpy as np

model = VitsModel.from_pretrained("facebook/mms-tts-nan")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-nan")

text = "some example text in the Chinese, Min Nan language"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

output_np = output.numpy()

print("Sampling rate:", model.config.sampling_rate)
standard_sampling_rate = 44100

if len(output_np.shape) == 1:
    # 将单声道数据转换为二维数组
    output_np = np.expand_dims(output_np, axis=-1)


# scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
scipy.io.wavfile.write("techno.wav", rate=standard_sampling_rate, data=output_np)

# text = """
# 曾经有一辆模型车在桌子上，它自诩拥有无与伦比的驾驶技巧。
# 一天，它向附近的水杯挑衅：“我才不害怕你！”
# """
# text = """
# 曾经有一辆模型车在桌子上，它自诩拥有无与伦比的驾驶技巧。
# 一天，它向附近的水杯挑衅：“我才不害怕你！”接着，它大摇
# 大摆地驶向杯子。可就在距离杯子仅剩一毫米的时候，模型车
# 突然紧张起来，因为它忘记了没有驾驶员！咚！模型车敲击在
# 杯子上，糗态百出。从此以后，每次骄傲的模型车依然自诩非
# 凡，但怎样驾驶都无济于事。因为无论多么强大的马力，没有
# 一个人执掌方向，车辆也只能在原地打转了。
# """
