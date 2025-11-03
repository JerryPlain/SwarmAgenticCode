TASK_MINI = '''You are given a few constraints on the existing schedule of each participant, the meeting duration, and possibly some preferences on the meeting time. 
Note there exists a solution that works with existing schedule of every participant. Please find a time that works for everyone's schedule and constraints.
Please provide the selected time in the format 'Weekday, HH:MM - HH:MM' (for example, 'Monday, 12:30 - 13:00').'''

FUNCTION_DESCRIPTION = """
The function is responsible for find a time that works for everyone's schedule and constraints by coordinating with various specialists. 
The function signature should be 'def forward(team):'.
The function returns a feasible time.
"""

TASK_OUTPUT_SCHEMA = None