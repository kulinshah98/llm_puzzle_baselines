import numpy as np
import pdb
from tensorflow.compat.v1 import gfile


def get_puzzles(args, file):
    path = args.folder + file
    with gfile.Open(path, "rb") as f:
        inputs_with_start_index = np.load(f)
    start_index = inputs_with_start_index[:, 0]

    inputs = np.delete(
        inputs_with_start_index[:, 1:], np.arange(81) * 4 + 3, axis=1)
    puzzles = np.zeros((len(inputs), 81), dtype=np.int8)
    
    for j in range(81):
        cell_id = inputs[:, 3 * j] * 9 + inputs[:, 3 * j + 1]
        puzzles[np.arange(len(inputs)), cell_id] = inputs[:, 3 * j + 2]

    # Inputs: 81 * 3 entries for each puzzle ordered according to the solver
    # Puzzles: 81 entries for each puzzle where 9 * i + j contains [i,j]th entry
    # Start index: number of given non empty cells in the puzzle.
    data = {"inputs": inputs, "puzzles": puzzles, "start_index": start_index}
    return data


def add_inputs(id, iter_start, iter_end, data):
    prompt = ""
    for i in range(iter_start, iter_end):
        # Row and column number are zero indexed in the dataset
        if i%3 == 0:
            prompt += "Row " + str(data["inputs"][id, i] + 1)
        elif i%3 == 1:
            prompt += " Column " + str(data["inputs"][id, i] + 1)
        elif i%3 == 2:
            prompt += " Value " + str(data["inputs"][id, i]) + "\n"

    return prompt


def get_puzzle_prompt(id, data):
    prompt = "Puzzle:\n\n"
    start_index = data['start_index'][id]
    prompt += add_inputs(id, iter_start=0, iter_end=3 * start_index, data=data)
    return prompt


def get_solution_prompt(id, data):
    prompt = "Solution:\n\n"
    start_index = data['start_index'][id]
    prompt += add_inputs(id, iter_start=3 * start_index, iter_end=3 * 81, data=data)
    return prompt


def get_parsed_response(response_list, args):
    
    num_response_list = []
    for i, response in enumerate(response_list):
        # Observe that in the examples on both sides of the row, column and value number, 
        # there is either a space or "\n" so to parse. We exploit it to parse the model's output.
        space_separated_response = response.replace("\n", " ").split(" ")

        # num_response list will contain row, column and value number hopefully for all the
        # empty cells in the puzzles.  
        num_response = []
        for word in space_separated_response:
            if word.isnumeric():
                num_response.append( int(word) )
        
        num_response_list.append(num_response)

    return num_response_list


def get_response_stats(parsed_responses, id, test_data, args):
    given_cells = np.zeros((9, 9))
    for i in range(test_data['start_index'][id]):
        given_cells[ 
            test_data['inputs'][id, 3 * i], 
            test_data['inputs'][id, 3 * i + 1] 
            ] = 1

    correct_response = np.zeros((len(parsed_responses), 9, 9))
    
    for i, response in enumerate(parsed_responses):
        for j in range(0, len(response), 3):
            
            if len(response) <= j+2:
                continue
            
            row = response[j] - 1
            col = response[j+1] - 1
            val = response[j+2]
            
            if row < 0 or row > 8:
                continue
            if col < 0 or col > 8:
                continue
            if val < 1 or val > 9:
                continue

            if given_cells[row, col] == 1:
                continue
            
            if test_data['puzzles'][id][9 * row + col] == response[j+2]:
                correct_response[i, row, col] = 1

            # print("R:", row, "C:", col, "V:", val,
            #       test_data['puzzles'][id][9 * row + col], 
            #       correct_response[i, row, col])        
    
    correct_fraction = correct_response.sum(axis=1).sum(axis=1) / (81 - test_data['start_index'][id])
    correct_fraction_in_any_response = np.sum(correct_response.sum(axis=0) > 0) / (81 - test_data['start_index'][id])
    
    return correct_fraction, correct_fraction_in_any_response
    





