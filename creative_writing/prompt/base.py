TASK_MINI = ''' Given 4 thematically unrelated sentences, write a coherent short narrative composed of 4 concise paragraphs. Each paragraph must end with one of the given sentences, in the provided order. Ensure the passage forms a unified story with a consistent tone, setting, and character perspective. Use the surrounding context to integrate each sentence naturally, maintaining emotional and narrative coherence throughout.'''

FUNCTION_DESCRIPTION = """
The function is responsible for writing a coherent passage by coordinating with various specialists. 
The function signature should be 'def forward(team):'.
The function returns a coherent passage.
"""

TASK_OUTPUT_SCHEMA = None