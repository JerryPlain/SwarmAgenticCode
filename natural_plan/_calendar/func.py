import json
import random
from langchain_core.prompts import PromptTemplate

from prompt.base import TASK_OUTPUT_SCHEMA
from prompt.write_forward import build_forward


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
    