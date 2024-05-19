import json 

from models import hf_model
from metrics import gpt_judge
from tasks import tldr 

task = tldr.Tldr(subset='comparisons', split='validation', shuffle=True, n=10)
df = task.dataset()
df['formatted_prompt'] = df.apply(task.formatter, axis=1)
model_a = hf_model.HFModel('EleutherAI/pythia-70m')
df['model_a_scrape'] = df['formatted_prompt'].apply(lambda x: model_a.predict(x))
model_b = hf_model.HFModel('EleutherAI/pythia-160m')
df['model_b_scrape'] = df['formatted_prompt'].apply(lambda x: model_b.predict(x))

judge = gpt_judge.GptJudge('gpt-4o')
judge_results = judge.judge(df['formatted_prompt'], df['model_a_scrape'], df['model_b_scrape'])
df.to_csv('results/test.csv')
with open('results/test.json', 'w') as f:
    json.dump(judge_results, f, ensure_ascii=False)

# Try a quick evaluation between two small pythia models.


# # model = hf_model.HFModel('CarperAI/openai_summarize_tldr_sft')
# print(model.decode_step('Hello! What is your name?'))

# judge = gpt_judge.GptJudge('gpt-4o')
# prompts = ['This is a test post about the election', 'THis is another test post about the super bowl']
# responses_a = ['oirnowigjogi', 'wrogijrgoij']
# responses_b = ['Test post', 'Test post']
# judge_results = judge.judge(prompts, responses_a, responses_b)
# print(judge_results)