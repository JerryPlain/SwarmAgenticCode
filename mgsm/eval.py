from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
# Prompt template for the coherence evaluation task. 
# Given passage.
# Output: JSON object with score.
SCORE_PROMPT = '''
Analyze the following passage and rate its coherence using an integer score from 1 to 10:
<passage>
{response}
<\passage>
'''

SCORE_SCHEMA = {
    "title": "Score",
    "description": "Coherence score.",
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": "Analyze the given passage and rate its coherence using an integer score from 1 to 10.",
        },
    },
    "required": ["score"]
}

PROBLEM_PROMPT = '''
For the task: {task}
You got the following passage: {response}
You are also providing its coherence score, ranging from 1 to 10, with 10 being the highest and 1 the lowest: {score}
Please help me point out the problems with the coherence of the current passage to improve its score.
Limit your answer in 50 words.
'''


base_prompt = """
You are given a detailed, step-by-step solution to a math problem. 
Your task is to extract the final numerical answer.

Return only the answer as a plain integer — no currency symbols, formatting (like bold or LaTeX), units, or explanation.

If the final answer is missing or ambiguous, respond with `None`.

Output format:
[integer] (e.g., 64)

Now extract the final answer from the following text:
---
{solution}
"""

FINAL_ANSWER_SCHEMA = {
    "title": "FinalAnswer",
    "description": "Extract the final integer answer from a math explanation.",
    "type": "object",
    "properties": {
        "answer": {
            "type": "integer",
            "description": "The final numerical answer extracted from the explanation. It must be a single integer with no units or formatting.",
        },
    },
    "required": ["answer"]
}


def extract_final_answer(llm, text):
    prompt = PromptTemplate(
        template=base_prompt,
        input_variables=["solution"],
    )
    chain = prompt | llm.with_structured_output(FINAL_ANSWER_SCHEMA)
    res = chain.invoke({"solution": text})
    if res["answer"] is None:
        return None
    return res['answer']
    

def evaluate(llm, response, answer):
    """True rating function. evaluate the coherence of the response. If the score is less than 10, return the problems with the response. Otherwise, return an empty string with the response.

    Args:
        llm (_type_): LLM model to be used.
        task (string): task description.
        response (string): response of the workflow function.

    Returns:
        (int, string): final_score, problem
    """
    extracted = extract_final_answer(llm, response)
    answer = answer.replace(",", "")
    if extracted is None:
        print(f"Warning: The extracted answer is None, the response is {response} and the answer is {answer}")
        return 0, "The extracted answer is None"
    try:
        score = int(extracted) == int(answer)
    except ValueError:
        print(f"Warning: The extracted answer is not an integer, the response is {extracted} and the answer is {answer}")
        score = 0

    problem = ""
    if score != 1:
        print(f"Warning: The score is not 1, the response is {extracted} and the answer is {answer}")
        problem = f"The list of correct answer is {answer}"
    return score, problem

def get_fitness(results):
    """calc the fitness of the results, which is the average score of the results

    Args:
        results (_type_): _description_

    Returns:
        int: average score of the results
    """
    scores = [result['score'] for result in results]
    fitness = float(sum(scores)) / len(scores)
    return fitness