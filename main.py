from models import hf_model
from evaluators import gpt_judge

# model = hf_model.HFModel('CarperAI/openai_summarize_tldr_sft')
model = hf_model.HFModel('EleutherAI/pythia-70m')
print(model.decode_step('Hello! What is your name?'))

judge = gpt_judge.GptJudge('gpt-4o')
prompts = ['This is a test post about the election', 'THis is another test post about the super bowl']
responses_a = ['oirnowigjogi', 'wrogijrgoij']
responses_b = ['Test post', 'Test post']
judge_results = judge.judge(prompts, responses_a, responses_b)
print(judge_results)
