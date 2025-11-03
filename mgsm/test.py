"""
Testing and Evaluation Script for MGSM Task
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


def execute(team_with_task, data, i, func, llm_eval):
    """Execute the team workflow on a single data point.
    
    Args:
        team_with_task: Team object with task
        data: Input data dict with 'inputs' and 'targets'
        i: Index of the data
        func: Forward function
        llm_eval: LLM for evaluation
        
    Returns:
        dict: Result with response and score
    """
    task_instance = data['inputs']
    team_with_task.reset_task(task_instance)
    
    times = 0
    while times < 3:
        try:
            res = func(team_with_task)
            break
        except Exception as e:
            team_with_task.reset_task(task_instance)
            print(f'Error in execution: {e}')
            times += 1
            if times >= 3:
                res = "Error: Failed after 3 attempts"
    
    # Evaluation 
    score, _ = evaluate(llm_eval, res, data['targets'])

    result = {
        "idx": i,
        "response": res,
        "target": data['targets'],
        "score": score,
    }
    return result


async def evaluate_particle(team, func, testset, llm_eval, save_dir, 
                            start_index=0, end_index=None, max_workers=16):
    """Evaluate a particle on the test dataset.
    
    Args:
        team: Team object
        func: Forward function
        testset: Test dataset
        llm_eval: LLM for evaluation
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
                executor, execute, team_with_task, data, i, func, llm_eval
            ))

        results = await asyncio.gather(*tasks)

    # Save results
    result_file = f'{save_dir}/results.jsonl'
    write_jsonl(result_file, results, 'w')
    print(f"Results saved to: {result_file}")
    
    # Calculate and print fitness
    fitness = get_fitness(results)
    print(f"\nFitness Score (Accuracy): {fitness:.4f}")
    
    return fitness


async def main(particle_idx=-1, model='gpt-4o-mini', eval_model='gpt-4o-mini',
               save_dir='results/test', start_index=0, end_index=None, 
               max_workers=16, test_start=128, test_end=928):
    """Main function to evaluate a particle on test dataset.
    
    Args:
        particle_idx: Index of particle to load from save.jsonl (default: -1, last particle)
        model: Model to use for team execution
        eval_model: Model to use for evaluation
        save_dir: Directory to save results
        start_index: Starting index in evaluation range
        end_index: Ending index in evaluation range (None = to end of test set)
        max_workers: Maximum number of worker threads
        test_start: Starting index for test dataset slice (default: 128)
        test_end: Ending index for test dataset slice (default: 928)
    """
    
    # Load test dataset (indices 128:928, 800 examples)
    datasets = get_all_examples()[test_start:test_end]
    
    print(f"Loaded {len(datasets)} test examples (indices {test_start}:{test_end})")
    
    # Setup models
    llm_role = ChatOpenAI(model=model, temperature=0.001)
    llm_eval = ChatOpenAI(model=eval_model, temperature=0.001)
    
    print(f"Using execution model: {model}")
    print(f"Using evaluation model: {eval_model}")
    
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
        team, func, datasets, llm_eval, save_dir,
        start_index, end_index, max_workers
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a saved particle on MGSM test dataset')
    parser.add_argument('--particle_idx', type=int, default=-1,
                        help='Index of particle to load from save.jsonl (default: -1, last particle)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Model to use for team execution (default: gpt-4o-mini)')
    parser.add_argument('--eval_model', type=str, default='gpt-4o-mini',
                        help='Model to use for evaluation (default: gpt-4o-mini)')
    parser.add_argument('--save_dir', type=str, default='results/test',
                        help='Directory to save results (default: results/test)')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting index in dataset (default: 0)')
    parser.add_argument('--end_index', type=int, default=None,
                        help='Ending index in dataset (default: None, to end)')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Maximum number of worker threads (default: 16)')
    parser.add_argument('--test_start', type=int, default=128,
                        help='Starting index for test dataset slice (default: 128)')
    parser.add_argument('--test_end', type=int, default=928,
                        help='Ending index for test dataset slice (default: 928, gives 800 examples)')
    
    args = parser.parse_args()
    
    asyncio.run(main(
        particle_idx=args.particle_idx,
        model=args.model,
        eval_model=args.eval_model,
        save_dir=args.save_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        max_workers=args.max_workers,
        test_start=args.test_start,
        test_end=args.test_end
    ))

