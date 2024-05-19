# Corresponding from the Conciseness version of the prompt from 
# https://cdn.jsdelivr.net/gh/LuYF-Lemon-love/susu-ChatGPT-papers/papers/07-DPO.pdf
LLM_JUDGE_TLDR_PROMPT = """Which of the following summaries does a better job of summarizing the most \
important points in the given forum post, without including unimportant or \
irrelevant details? A good summary is both precise and concise.
Post:
{post}
Summary A:
{summary_a}
Summary B:
{summary_b}
FIRST provide a one-sentence comparison of the two summaries, explaining which \
you prefer and why. SECOND, on a new line, state only "A", "B", or "TIE" to indicate your \
choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A", "B", or "TIE">"""
