TASK_MINI = '''Given a grade-school level math word problem in English, provide a detailed, step-by-step solution that demonstrates clear reasoning leading to the correct final answer. The solution should be concise, logically structured, and use appropriate mathematical operations. Ensure that each step is explained to showcase the reasoning process, culminating in the final answer presented on a separate line in the format: "Answer: [final answer]". The final answer must be a single integer, with no units, punctuation, or explanation after it—just the word Answer: followed by the number. Do not include any text or commentary after this final answer line.'''


FUNCTION_DESCRIPTION = """
The function is responsible for writing a coherent passage by coordinating with various specialists. 
The function signature should be 'def forward(team):'.
The function returns a coherent passage.
"""

TASK_OUTPUT_SCHEMA = None