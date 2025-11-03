import json
import random
from langchain_core.prompts import PromptTemplate

from prompt.base import TASK_OUTPUT_SCHEMA
from prompt.write_forward import build_forward

LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem.

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন।

{input}""",
    "de": """Löse dieses Mathematikproblem.

{input}""",
    "es": """Resuelve este problema matemático.

{input}""",
    "fr": """Résolvez ce problème de mathématiques.

{input}""",
    "ja": """この数学の問題を解いてください。

{input}""",
    "ru": """Решите эту математическую задачу.

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu.

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి.

{input}""",
    "th": """แก้ปัญหาคณิตศาสตร์นี้

{input}""",
    "zh": """解决这个数学问题。

{input}"""
}

ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


def read_jsonl(file_path) -> list:
    """Read a JSON file and convert it to a list of dictionaries.

    Args:
        file_path (string): file path of the JSON file.

    Returns:
        list: list of dictionaries.
    """
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            jsondict = json.loads(jsonstr)
            data.append(jsondict)
    return data

def write_jsonl(file_path, data, mode):
    """write a list of dictionaries to a JSON file.

    Args:
        file_path (string): file path of the JSON file.
        data (list(dict)): data to be written to the file, each element is a dictionary.
        mode (char): mode of the file, 'w' for write, 'a' for append.
    """
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_forward(llm, logger, roles, workflow):
    """Generate the functions that represent the workflow of the roles.

    Args:
        llm (string): LLM model to be used.
        logger (logger): a logger to record the process.
        roles (role): roles in the team.
        workflow (_type_): workflow of the roles.

    Returns:
        _type_: A function that represents the workflow of the roles.
    """
    try:
        next_solution = build_forward(llm, logger, roles, workflow)
    except Exception as e:
        print("During LLM generate new solution:")
        print(e)
    return next_solution

def get_lang_examples(lang:str = 'en') -> list[dict[str, str]]:
    fpath = f'data/mgsm_{lang}.tsv'
    examples = []
    with open(fpath, mode='r', encoding='utf-8') as f:
        for line in f:
            inputs, targets = line.strip().split("\t")
            if "." in targets:
                raise ValueError(f"targets {targets} contains a decimal point.")
            # targets = int(targets.replace(",", ""))
            examples.append({"inputs": LANG_TO_INSTRUCTIONS[lang].format(input=inputs), "targets": targets, "lang": lang})
    return examples

def get_all_examples() -> list[dict[str, str]]:
    examples = []
    for lang in ALL_LANGUAGES:
        # if lang != "en":
        #     continue
        examples += get_lang_examples(lang)
    return examples

def set_forward(next_solution):
    """
    Generate a function from the given string of Python code to be executed by team.

    Args:
        next_solution (str): A string of Python code that defines a single function or callable object.

    Returns:
        Callable: The function or callable object defined by the input code.

    Raises:
        AssertionError: If the executed code defines more than one or fewer than one name, 
            or if the defined name is not callable.
    """
    namespace = {}
    exec(next_solution, globals(), namespace)
    names = list(namespace.keys())
    # only one variable should be defined, which is the team.
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    return func

def load_particles(idx: int):
    """Load particles from the save file.

    Args:
        idx (int): index of the particles to be loaded.

    Returns:
        _type_: particles.
    """
    archive = read_jsonl('save.jsonl')
    particles = archive[idx]['archive']
    return particles

def save_particles(particles):
    archive = []
    for particle in particles:
        archive.append({"team": particle.best_position[0], "code": particle.best_position[1], "score": particle.best_fitness})
    write_jsonl("save.jsonl", [{"archive": archive}], 'a')
    