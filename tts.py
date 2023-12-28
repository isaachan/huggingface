from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("曾经有一辆模型车在桌子上，它自诩拥有无与伦比的驾驶技巧。一天，它向附近的水杯挑衅：“我才不害怕你！”接着，它大摇大摆地驶向杯子。可就在距离杯子仅剩一毫米的时候，模型车突然紧张起来，因为它忘记了没有驾驶员！咚！模型车敲击在杯子上，糗态百出。从此以后，每次骄傲的模型车依然自诩非凡，但怎样驾驶都无济于事。因为无论多么强大的马力，没有一个人执掌方向，车辆也只能在原地打转了。", 
                     forward_params={"speaker_embeddings": speaker_embedding})

sf.write("story.wav", speech["audio"], samplerate=speech["sampling_rate"])
