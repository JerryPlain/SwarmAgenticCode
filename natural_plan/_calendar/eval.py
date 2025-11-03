import re


def hour_to_num(hr_str):
    return float(hr_str.split(':')[0]) + (
        0.5 if hr_str.split(':')[1] == '30' else 0.0
    )


def _parse_response(response: str):
    """Parse the response.

    Returns a parsed suggested meeting time in (day, start_hour, end_hour).

    Args:
    response: Raw response from the model.

    Returns:
    A tuple of (day, start_hour, end_hour).
    """
    time_strs = re.findall(r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+', response)
    if not time_strs:
        return '', -1, -1
    # If multiple matches are found, return the first one.
    time_str = time_strs[0]
    day, hour_str = (
        time_str.split(',')[0].strip(),
        time_str.split(',')[1].strip(),
    )
    start_hour, end_hour = (
        hour_str.split('-')[0].strip(),
        hour_str.split('-')[1].strip(),
    )
    return day, hour_to_num(start_hour), hour_to_num(end_hour)


def compute_solve_rate(responses: list[str], solutions: list[str]):
    """Computes solve rate by comparing model responses to golden solutions.

    Args:
    responses: A list of model responses.
    solutions: The corresponding list of golden solutions for the same tasks.

    Returns:
    A scalr solve rate.
    """
    solved_count = 0

    for r, s in zip(responses, solutions):
        r_day, r_start_hour, r_end_hour = _parse_response(r)
        s_day, s_start_hour, s_end_hour = _parse_response(s)
        if (
            r_day == s_day
            and r_start_hour == s_start_hour
            and r_end_hour == s_end_hour
        ):
            solved_count += 1
    return float(solved_count) / len(responses)

def evaluate(data, response):
    """Compare the response with the golden plan and return the score and problem.

    Args:
        data (_type_): golden plan
        response (_type_): A list of model responses.

    Returns:
        _type_: A tuple of (score, problem). 1.0 if the response is correct, 0.0 otherwise.
    """
    solved = compute_solve_rate([response], [data['golden_plan']])
    if solved:
        return 1.0, ''
    else:
        time_strs = re.findall(r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+', data['golden_plan'])
        problem = f'''The time planned is incorrect, while The correct time is "{time_strs}".'''
        return 0.0, problem

def get_fitness(results):
    scores = [result['score'] for result in results]
    fitness = sum(scores) / len(scores)
    return fitness