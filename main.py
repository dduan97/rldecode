from models import hf_model 

# model = hf_model.HFModel('CarperAI/openai_summarize_tldr_sft')
model = hf_model.HFModel('EleutherAI/pythia-70m')
print(model.decode_step('Hello! What is your name?'))