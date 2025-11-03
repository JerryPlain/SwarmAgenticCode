import json
import random
from prompt.base import TASK_OUTPUT_SCHEMA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompt.write_forward import build_forward


EXTRACT_PALN_TEMPLATE_1 = '''Please assist me in extracting valid information from a given text and reconstructing it in the required format.

**Required:**
- Your response should start with 'SOLUTION:'.
- The time format must be "9:00AM" and must NOT be "9:00 AM".
- You can only use the following four sentence structures:
    1. "You start at {{location}} at {{start time}}."
    2. "You travel to {{location}} in {{time taken}} minutes and arrive at {{arrival time}}."
    3. "You wait until {{end time}}."
    4. "You meet {{person name}} for {{time taken}} minutes from {{start time}} to {{end time}}."

Here is an example:
<example>
SOLUTION: 
You start at Russian Hill at 9:00AM.
You travel to Marina District in 7 minutes and arrive at 9:07AM.
You wait until 3:45PM.
You meet James for 75 minutes from 3:45PM to 5:00PM.
</example>

Now, giving the following text:
<text>
{text}
</text>

Please extract valid information from the given <text> and reconstruct it in the same solution format as demonstrated in the <example>.
'''


EXTRACT_PALN_TEMPLATE_2 = '''Please assist me in extracting valid information from a given text and reconstructing it in the required format.

Here is an example:
<example>
Text:
SOLUTION:
You start at Russian Hill at 9:00AM.
You travel to Marina District in 7 minutes and arrive at 9:07AM.
You wait until 3:45PM.
You meet James for 75 minutes from 3:45PM to 5:00PM.

Answer:
[
    {{
        "location": "Russian Hill",
        "person_name": "N/A",
        "start_time": "9:00AM"
    }},
    {{
        "location": "Marina District",
        "person_name": "N/A",
        "start_time": "9:07AM"
    }},
    {{
        "location": "Marina District",
        "person_name": "James",
        "start_time": "3:45PM"
    }}
]
</example>

Now, giving the following text:
<text>
{text}
</text>

Please extract valid information from the given <text> and reconstruct it in the same solution format as demonstrated in the <example>.
'''


def read_jsonl(file_path) -> list:
    "read jsonl file into list"
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsondict = json.loads(jsonstr)
            data.append(jsondict)
    return data


def write_jsonl(file_path, data, mode):
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_dataset(path, shuffle=False, seed_index=0):
    dataset = read_jsonl(path)
    if shuffle:
        random.seed(seed_index)
        random.shuffle(dataset)
    return dataset


def get_forward(llm, logger, roles, workflow):
    try:
        next_solution = build_forward(llm, logger, roles, workflow)
    except Exception as e:
        print("During LLM generate new solution:")
        print(e)
    return next_solution


def set_forward(next_solution):
    namespace = {}
    exec(next_solution, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    return func


def load_particles(idx: int):
    archive = read_jsonl('save.jsonl')
    particles = archive[idx]['archive']
    return particles


def save_particles(particles):
    archive = []
    for particle in particles:
        archive.append({"team": particle.best_position[0], "code": particle.best_position[1], "score": particle.best_fitness})
    write_jsonl("save.jsonl", [{"archive": archive}], 'a')
    
