TASK_MINI = '''You want to meet as many friends as possible in the same day. 
Each friend will appear in a specific place within a specific time period.
You'd like to meet each friend for a minimum time requirement.
You will get the time it takes to travel to different places.
Make a optimal meeting plan to achieve the goal.
'''

FUNCTION_DESCRIPTION = """
The function is responsible for find a meeting plan under specific time constraints and travel durations by coordinating with various specialists. 
The function signature should be 'def forward(team):'.
The function returns a feasible meeting plan.
"""

TASK_OUTPUT_SCHEMA = {
    "title": "Plan",
    "description": "Detailed meeting plan.",
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "description": "Meeting plan for the day.",
            "items": {
                "type": "object",
                "properties": {
                    "location": {
                         "type": "string",
                         "description": "Current location. If you travel to a new place, then the new place. If meet a person, then the place stay unchanged.",
                    },
                    "person_name": {
                         "type": "string",
                         "description": "Person to meet. 'N/A' if no person to meet.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time. The time format must be '9:00AM' and must NOT be '9:00 AM'.",
                    },
                },
                "required": ["location", "person_name", "start_time"],
            }
        },
    },
    "required": ["plan"],
}