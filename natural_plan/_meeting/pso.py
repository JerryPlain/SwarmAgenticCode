"""
PSO (Particle Swarm Optimization) for Meeting Planning Task
"""

from langchain_openai import ChatOpenAI

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import argparse
import copy
import json
import os
from tqdm import tqdm
from func import *
from role import Team
from eval import evaluate, get_fitness
from logger import setup_logger, log, log_all

from prompt.team_update import update_team
from prompt.velocity_init import initialize_velocity
from prompt.velocity_update import update_velocity
from prompt.failure_identify import identify_failure
from prompt.failure_improve import improve_failure
from prompt.best_global import reflect_from_global_best
from prompt.best_personal import reflect_from_personal_best
from prompt.feedback_give import give_feedback
from prompt.feedback_summarize import summarize_feedback


class Particle:
    def __init__(self, position, logger, llm, save_dir='results', max_workers=None):
        self.position = position
        self.velocity = None
        self.fitness = 0.0
        self.evaluation = None

        self.best_position = None
        self.best_fitness = 0.0  

        self.logger = logger
        self.llm = llm
        self.save_dir = save_dir
        self.max_workers = max_workers
        self.fitness_history = []  

    async def evaluate(self, dataset, iter, i_pos):
        team = self.position[0]
        code = self.position[1]
        func = set_forward(code)

        def execute(team_with_task, data, evaluations, i, batch_logs):
            logs = [(f'Iter {iter} - Data {i}', '\n# Roles Message', None)]
            
            task_instance = data["prompt_0shot"]
            team_with_task.reset_task(task_instance)
            times = 0
            while times < 3:
                try:
                    res = func(team_with_task)
                    logs.extend(team_with_task.logs)
                    break
                
                except Exception as e:
                    logs = [(f'Iter {iter} - Data {i}', '\n# Roles Message', None)]
                    team_with_task.reset_task(task_instance)
                    print(f'{e}')
                    times += 1

            score, problem = evaluate(data, res)
            logs.append((f'Iter {iter} - Data {i} - Correct', f'''{bool(score)}\n{data['golden_plan']}''', None))

            if score == 0.0:
                evaluation = give_feedback(self.llm, self.logger, task_instance, team_with_task.patch_result_and_workflow(), problem)
                evaluations.append(evaluation)
            batch_logs.append(logs)
            
            result = {
                    "response": res,
                    "score": score,
            }
            return result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_running_loop()
            tasks = []
            evaluations = []
            batch_logs = []
            for i, data in enumerate(dataset):
                team_with_task = team.deepcopy() 
                tasks.append(loop.run_in_executor(executor, execute, team_with_task, data, evaluations, i, batch_logs))

            results = await asyncio.gather(*tasks)
            for logs in batch_logs:
                log_all(self.logger, logs)

        os.makedirs(self.save_dir, exist_ok=True)
        write_jsonl(f'{self.save_dir}/results-{iter}-{i_pos}.jsonl', results, 'w')

        self.fitness = get_fitness(results)
        log(self.logger, 'Pass Rate', self.fitness)

        self.fitness_history.append(self.fitness)               
        if self.fitness >= self.best_fitness:
            self.best_position = (self.position[0].save_into_dict(), self.position[1])
            self.best_fitness = self.fitness

        evaluations = '\n'.join(f"**Feedback {i+1}:**\n{item}\n" for i, item in enumerate(evaluations))
        self.evaluation = summarize_feedback(self.llm, self.logger, evaluations, json.dumps(team.save_into_dict(), indent=4))
 
    def update_velocity(self, global_best_position):
        team = json.dumps(self.position[0].save_into_dict(), indent=4)
        global_best_team = json.dumps(global_best_position[0], indent=4)
        personal_best_team = json.dumps(self.best_position[0], indent=4)
        
        evaluation = self.evaluation

        if team == personal_best_team:
            p_best = None
        else:
            p_best = reflect_from_personal_best(self.llm, self.logger, team, evaluation, personal_best_team)
            
        if team == global_best_team:
            g_best = None
        else:
            g_best = reflect_from_global_best(self.llm, self.logger, team, evaluation, global_best_team)
        
        if self.velocity is None:
            velocity = initialize_velocity(self.llm, self.logger, team, evaluation)
            if team == personal_best_team and team == global_best_team:
                self.velocity = velocity
            else:
                self.velocity = update_velocity(self.llm, self.logger, team, velocity, g_best, p_best)
        else:
            failures = identify_failure(self.llm, self.logger, evaluation, self.velocity)
            velocity = improve_failure(self.llm, self.logger, team, failures)
            if team == personal_best_team and team == global_best_team:
                clean_velocity = [{k: v for k, v in item.items() if k != "Failed Adjustment"} for item in velocity]
                self.velocity = clean_velocity
            else:
                self.velocity = update_velocity(self.llm, self.logger, team, velocity, g_best, p_best)

    def update_position(self):
        team = self.position[0]
        new_team = update_team(self.llm, self.logger, team.to_str(), team.workflow, self.velocity)
        team.update(new_team)
        new_code = get_forward(self.llm, self.logger, team.to_str(), team.workflow)
        self.position = (team, new_code)


def initialize(settings, llm_role, llm_eval, model, save_dir='results', max_workers=None):
    particles = []
    for i, item in enumerate(settings): 
        logger = setup_logger(i)
        llm = ChatOpenAI(model=model, temperature=item)
        team = Team(llm=llm_role, logger=logger)
        team.init(llm=llm)
        code = get_forward(llm_eval, logger, team.to_str(), team.workflow) 
        particle = Particle(
            position = (team, code),
            logger = logger,
            llm = llm_eval,
            save_dir = save_dir,
            max_workers = max_workers
        )
        particles.append(particle)
    return particles


def initialize_with_archive(idx: int, llm_role, llm_eval, save_dir='results', max_workers=None):
    particles = []
    archives = load_particles(idx)
    for i, archive in enumerate(archives): 
        logger = setup_logger(i)
        team = Team(llm=llm_role, logger=logger)
        team.update(archive['team'])
        code = archive['code'] 
        particle = Particle(
            position = (team, code),
            logger = logger,
            llm = llm_eval,
            save_dir = save_dir,
            max_workers = max_workers
        )
        particles.append(particle)
    return particles


def update_global_best(particles, global_best_position, global_best_fitness):
    g_best_position = global_best_position
    g_best_fitness = global_best_fitness
    
    for p in particles:
        if p.fitness >= g_best_fitness:
            g_best_position = p.best_position
            g_best_fitness = p.fitness
    
    return g_best_position, g_best_fitness


async def main(max_iteration=10, settings=None, model='gpt-4o-mini', max_workers=None, save_dir='results',
               dataset_path='../data/meeting_planning.json', sample_step=100):
    """Main function to run PSO optimization.
    
    Args:
        max_iteration (int): Maximum number of iterations (default: 10)
        settings (list): Temperature settings for each particle (default: [0.2, 0.4, 0.6, 0.8, 1])
        model (str): Model name to use (default: 'gpt-4o-mini')
        max_workers (int): Maximum number of worker threads (default: None, uses system default)
        save_dir (str): Directory to save results (default: 'results')
        dataset_path (str): Path to dataset file (default: '../data/meeting_planning.json')
        sample_step (int): Sampling step for dataset (default: 100, takes every 100th item)
    """
    # Load dataset
    train_set = load_dataset(path=dataset_path)
    dataset = [v for v in train_set[0].values()]
    dataset = dataset[::sample_step]
    print(f"Loaded {len(dataset)} examples (sampled every {sample_step} items)")

    iter = 0
    if settings is None:
        settings = [0.2, 0.4, 0.6, 0.8, 1]

    particles = []
    global_best_position = None
    global_best_fitness = 0.0
    global_best_trend = []

    # Hyper Parameter
    llm_role = ChatOpenAI(model=model, temperature=0.001)
    llm_eval = ChatOpenAI(model=model, temperature=0.001)

    particles = initialize(settings, llm_role, llm_eval, model, save_dir, max_workers)

    for iter in tqdm(range(max_iteration), desc="Iteration", position=0):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            evaluate_tasks = []
            for i, p in enumerate(particles):
                evaluate_tasks.append(p.evaluate(dataset, iter, i))
            awaitables = asyncio.as_completed(evaluate_tasks)
            for _ in tqdm(awaitables, desc="Particles Evaluate", total=len(particles), position=1):
                await _

            global_best_position, global_best_fitness = update_global_best(particles, global_best_position, global_best_fitness)
            global_best_trend.append(copy.deepcopy(global_best_fitness))

            if iter < max_iteration - 1:
                velocity_futures = []
                for p in particles:
                    velocity_futures.append(executor.submit(p.update_velocity, global_best_position))
                for future in tqdm(as_completed(velocity_futures), desc="Update Velocity", total=len(particles), position=2):
                    future.result()

                position_futures = []
                for p in particles:
                    position_futures.append(executor.submit(p.update_position))
                for future in tqdm(as_completed(position_futures), desc="Update Position", total=len(particles), position=2):
                    future.result()

    save_particles(particles)
    
    print(f"\nOptimization completed!")
    print(f"Global best fitness: {global_best_fitness}")
    print(f"Global best trend: {global_best_trend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PSO Optimization for Meeting Planning Task')
    parser.add_argument('--max_iteration', type=int, default=10, 
                        help='Maximum number of iterations (default: 10)')
    parser.add_argument('--settings', type=float, nargs='+', default=None,
                        help='Temperature settings for each particle (default: [0.2, 0.4, 0.6, 0.8, 1])')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model name to use (default: gpt-4o-mini)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of worker threads (default: None, uses system default)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--dataset', type=str, default='../data/meeting_planning.json',
                        help='Path to dataset file')
    parser.add_argument('--sample_step', type=int, default=100,
                        help='Sampling step for dataset (default: 100)')
    
    args = parser.parse_args()
    
    asyncio.run(main(
        max_iteration=args.max_iteration, 
        settings=args.settings, 
        model=args.model,
        max_workers=args.max_workers,
        save_dir=args.save_dir,
        dataset_path=args.dataset,
        sample_step=args.sample_step
    ))

