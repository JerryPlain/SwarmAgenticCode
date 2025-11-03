"""
Testing and Evaluation Script for Trip Planning Task
Evaluate a saved particle on test dataset
"""

from langchain_openai import ChatOpenAI

import asyncio
from concurrent.futures import ThreadPoolExecutor

import argparse
import json
import os
from tqdm import tqdm
from func import *
from role import Team
from logger import setup_logger, log
from eval import evaluate, get_fitness


def execute(team_with_task, data, i, func):
    """Execute the team workflow on a single data point.
    
    Args:
        team_with_task: Team object with task
        data: Input data dict
        i: Index of the data
        func: Forward function
        
    Returns:
        dict: Result with response and score
    """
    task_description = data["prompt_0shot"]
    team_with_task.reset_task(task_description)
    
    res = func(team_with_task)

    # Evaluation 
    score, _ = evaluate(data, res)

    result = {
        "idx": i,
        "response": res,
        "score": score,
    }
    return result


async def evaluate_particle(team, func, testset, save_dir, 
                            start_index=0, end_index=None, max_workers=16):
    """Evaluate a particle on the test dataset.
    
    Args:
        team: Team object
        func: Forward function
        testset: Test dataset
        save_dir: Directory to save results
        start_index: Starting index in dataset
        end_index: Ending index in dataset (None = to end)
        max_workers: Maximum number of worker threads
        
    Returns:
        float: Average fitness score
    """
    if end_index is None:
        end_index = len(testset)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Evaluating on indices {start_index} to {end_index} ({end_index - start_index} examples)")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_running_loop()
        tasks = []
        for i, data in enumerate(testset[start_index:end_index]):
            team_with_task = team.deepcopy() 
            tasks.append(loop.run_in_executor(
                executor, execute, team_with_task, data, i, func
            ))

        results = await asyncio.gather(*tasks)

    # Save results
    result_file = f'{save_dir}/results.jsonl'
    write_jsonl(result_file, results, 'w')
    print(f"Results saved to: {result_file}")
    
    # Calculate and print fitness
    fitness = get_fitness(results)
    print(f"\nFitness Score: {fitness:.4f}")
    
    return fitness


def collect_scores_and_average(folder_path):
    """Collect all scores from JSONL files in a folder and compute average.
    
    Args:
        folder_path: Path to folder containing JSONL result files
        
    Returns:
        float: Average score
    """
    total_score = 0
    score_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'score' in data and isinstance(data['score'], (int, float)):
                            total_score += data['score']
                            score_count += 1
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in file: {filename}")

    if score_count == 0:
        print("No valid scores found.")
        return None

    average_score = total_score / score_count
    print(f"Total Score: {total_score}")
    print(f"Number of Scores: {score_count}")
    print(f"Average Score: {average_score:.4f}")
    return average_score


async def main(particle_idx=-1, model='gpt-4o-mini', 
               save_dir='results/test', start_index=0, end_index=None, 
               max_workers=16, dataset_path='../data/trip_planning.json',
               sample_step=10, aggregate_folder=None):
    """Main function to evaluate a particle on test dataset.
    
    Args:
        particle_idx: Index of particle to load from save.jsonl (default: -1, last particle)
        model: Model to use for team execution
        save_dir: Directory to save results
        start_index: Starting index in dataset
        end_index: Ending index in dataset (None = to end)
        max_workers: Maximum number of worker threads
        dataset_path: Path to dataset file
        sample_step: Sampling step for dataset (default: 10)
        aggregate_folder: If provided, aggregate scores from this folder instead of testing
    """
    
    # If aggregating existing results
    if aggregate_folder:
        print(f"Aggregating results from: {aggregate_folder}")
        collect_scores_and_average(aggregate_folder)
        return
    
    # Load dataset
    train_set = load_dataset(path=dataset_path)
    testset = [v for v in train_set[0].values()]
    testset = testset[::sample_step]
    
    print(f"Loaded {len(testset)} test examples (sampled every {sample_step} items)")
    
    # Setup model
    llm_role = ChatOpenAI(model=model, temperature=0.001)
    
    print(f"Using execution model: {model}")
    
    # Load particle
    logger = setup_logger(9)
    team = Team(llm_role, logger)
    
    try:
        particles = load_particles(particle_idx)
    except FileNotFoundError:
        print(f"Error: save.jsonl file not found. Please run training first.")
        return
    except IndexError:
        print(f"Error: Particle index {particle_idx} out of range in save.jsonl")
        return
    except Exception as e:
        print(f"Error loading particles: {e}")
        return
    
    if not particles:
        print(f"Error: No particles found at index {particle_idx}")
        return
    
    try:
        team_dict = particles[0]['team']
        code = particles[0]['code']
    except (IndexError, KeyError) as e:
        print(f"Error: Invalid particle data format - {e}")
        return
    
    team.update(team_dict)
    func = set_forward(code)
    
    log(logger, f'Particle {particle_idx}', f'''{team}\n\n{code}''')
    print(f"Loaded particle {particle_idx}")
    
    # Run evaluation
    await evaluate_particle(
        team, func, testset, save_dir,
        start_index, end_index, max_workers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a saved particle on Trip Planning test dataset')
    parser.add_argument('--particle_idx', type=int, default=-1,
                        help='Index of particle to load from save.jsonl (default: -1, last particle)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model to use for team execution (default: gpt-4o-mini)')
    parser.add_argument('--save_dir', type=str, default='results/test',
                        help='Directory to save results (default: results/test)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index in dataset (default: 0)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='Ending index in dataset (default: None, to end)')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Maximum number of worker threads (default: 16)')
    parser.add_argument('--dataset', type=str, default='../data/trip_planning.json',
                        help='Path to dataset file')
    parser.add_argument('--sample_step', type=int, default=10,
                        help='Sampling step for dataset (default: 10)')
    parser.add_argument('--aggregate_folder', type=str, default=None,
                        help='If provided, aggregate scores from this folder instead of testing')
    
    args = parser.parse_args()
    
    asyncio.run(main(
        particle_idx=args.particle_idx,
        model=args.model,
        save_dir=args.save_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        max_workers=args.max_workers,
        dataset_path=args.dataset,
        sample_step=args.sample_step,
        aggregate_folder=args.aggregate_folder
    ))

