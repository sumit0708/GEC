from grammarCorrector import Gramformer
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)


gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

incorrect_sentences = [
    "Delivering an impressive presentation in a limited time is challenging but I feel I could overcome it with a lot of practice",
    "Developing a framework for allocating strategies and recommendations as well as establishing business networks",
    "Extract insights from literature that are applicable and were useful to my make a difference project",
    "After today’s class I wil+C2+D2+E2+D2",
    "Apply theory to my make a difference project to provide clarity and logic to my make a difference project !!!",
    "what be the reason for everyone leave the company",
    "I am go to work on my group work skills"
]   

for inf_sentence in incorrect_sentences:
    corrected_sent = gf.correct(inf_sentence, max_candidates=1)
    print("[Input] ", inf_sentence)
    for corrected_sent in corrected_sent:
      print("[Predicted Sentence] ",corrected_sent)
    print("-" *100)
