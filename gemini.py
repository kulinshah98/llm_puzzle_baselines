
import argparse
import os
import pdb
import numpy as np
import re

import google.generativeai as genai

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--folder', type=str, default="/scratch/cluster/kulin/sudoku/data/")
parser.add_argument('--train_data_file', type=str, default="ordered-sudoku-3m-wo-random-incorrect-w-strategies-candidates-train.npy")
parser.add_argument('--test_data_file', type=str, default="ordered-sudoku-3m-wo-random-incorrect-w-strategies-candidates-test.npy")
parser.add_argument('--test_size', type=int, default=10)
parser.add_argument('--num_response', type=int, default=3)
args = parser.parse_args()
print(args)


np.random.seed(args.seed)

train_data = utils.get_puzzles(args, args.train_data_file)
test_data = utils.get_puzzles(args, args.test_data_file)


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')

# For each test puzzle, the code generates {args.num_response} responses
# and then calculates correct fraction of empty cells for that puzzle for each response.
# max_correct_fraction_list contains maximum of correct fraction of all these responses. 
max_correct_fraction_list = []

# The correct_fraction_in_any_response_list is fraction of empty cells for which atleast one of the {args.num_response} 
# response contains correct answer. 
correct_fraction_in_any_response_list = []

# If any of the response completely correctly solved the puzzle or not. 
complete_puzzle_num = []

for test_ind in range(args.test_size):
    prompt = "## Following two are examples of sudoku puzzles and its solution. ##\n\n\n"
        
    for i in range(2):
        prompt += "# Example " + str(i + 1) + "#\n\n"

        # Randomly selects id of training puzzle
        id = np.random.randint(len(train_data['inputs']))
        prompt += utils.get_puzzle_prompt(id, train_data)
        prompt += utils.get_solution_prompt(id, train_data)
        
    id = np.random.randint(len(test_data['inputs']))
    prompt += "\n\nSolve the following Sudoku puzzle and provide the solution in the form of the above example.\n\n"
    prompt += utils.get_puzzle_prompt(id, test_data)
    prompt += "Solution:\n\n"

    response_list = []
    for t in range(args.num_response):
        response = model.generate_content(prompt)
        response_list.append(response.text)

    parsed_responses = utils.get_parsed_response(response_list, args)

    correct_fraction, correct_fraction_in_any_response = utils.get_response_stats(
        parsed_responses, id, test_data, args
        )
    
    max_correct_fraction_list.append(correct_fraction.max())
    correct_fraction_in_any_response_list.append(correct_fraction_in_any_response)
    complete_puzzle_num.append( correct_fraction.max() > 0.99999 )
    


print("\n\nAvg correct fraction:", np.array(max_correct_fraction_list).mean(), 
      "Avg correct fraction in any of the response:", np.array(correct_fraction_in_any_response_list).mean(), 
      "Average correct complete puzzle accuracy:", np.array(complete_puzzle_num).mean())