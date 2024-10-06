import heapq

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.g = 0  # Acutal cost to reach current node 
        self.h = 0  # Heuristic cost 
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f  # To supportt heap queue comparison
    
class BlockWorldAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, initial_state, goal_state):
        # Create an open list (priority queue) to keep track of nodes 
        open_list = []
        
        initial_node = Node(initial_state)
        initial_node.h = heuristic(initial_state, goal_state)
        initial_node.f = initial_node.h

        # Add the start node to the open list
        heapq.heappush(open_list, initial_node)
        
        # Create a set (closed list) to store the hash of processed nodes
        closed_list = set()

        try_n = 0
        # Keep expanding nodes until there are no more nodes or the goal is found
        while open_list and try_n < 500000:
            try_n += 1
            # Pop the node with the lowest f score from the open list
            current_node = heapq.heappop(open_list)
            
            # Check if the goal state is reached
            if normalize_state(current_node.state) == normalize_state(goal_state):
                # If the goal is found, reconstruct and return the path
                return reconstruct_path(current_node)
            
            # Mark the current node as processed
            closed_list.add(str(current_node.state))
            
            # Generate neighbor nodes of the current node
            for neighbor, move in get_neighbors(current_node):
                # Skip nodes that have already been processed
                if str(neighbor.state) in closed_list: continue
                
                # Calculate the cost from start to neighbor through current node
                tentative_g = current_node.g + 1  # Assume cost of all actions is 1
                
                # Update neighbor node's scores and parent 
				# if the new path is better, or if neighbor is unvisited
                if tentative_g < neighbor.g or neighbor.g == 0:
                    neighbor.g = tentative_g  # Set new g score
                    neighbor.h = heuristic(neighbor.state, goal_state)  # Calculate heuristic score
                    neighbor.f = neighbor.g + neighbor.h  # Set new total cost f
                    neighbor.parent = current_node  # Set current node as neighbor's parent
                    neighbor.move = move  # Store the move that led to the neighbor
                    heapq.heappush(open_list, neighbor)  # Add neighbor to the open list for further processing

        # If no solution is found, return None
        return None
		
def normalize_state(state):
    return tuple(sorted(tuple(stack) for stack in state))

def get_block_relationships(state):
    # This function generates a dictionary indicating the block relationships
	# (top and bottom) for each block in a given state.
    relationships = {}  # Initialize an empty dictionary to store the relationships
    for stack in state:  # Iterate through each stack in the state
        for i, block in enumerate(stack):  # Iterate through each block in the stack
            # Determine the block on top, if any
            top_block = stack[i + 1] if i + 1 < len(stack) else None
            # Determine the block below, or 'Table' if it's the bottom block
            bottom_block = stack[i - 1] if i > 0 else 'Table'
            # Store the relationships in the dictionary
            relationships[block] = [top_block, bottom_block]
    return relationships  # Return the relationships dictionary

def trace_stack(block, current_relationships, goal_relationships):
    # This function traces a stack from the given block upwards, returning a list of blocks in order.
    correct_count = 0
    if current_relationships[block][1] != 'Table': 
        return 0
    
    correct_count += 1
    while goal_relationships[block][0] and goal_relationships[block][0] == current_relationships[block][0]:
        
        correct_count += 1
        block = goal_relationships[block][0]
    return correct_count

def heuristic(current_state, goal_state):
    current_relationships = get_block_relationships(current_state)
    goal_relationships = get_block_relationships(goal_state)
    
    # Identify the blocks on the table in the goal state.
    goal_table_blocks = [block for block, relations in goal_relationships.items() if relations[1] == 'Table']

    correct_count = 0
    for goal_table_block in goal_table_blocks:

        correct_count_stack = trace_stack(goal_table_block, current_relationships, goal_relationships)
        correct_count += correct_count_stack

    mismatch = len(current_relationships) - correct_count
    
    return mismatch * 2

def remove_empty_stack(state):
    return [stack for stack in state if len(stack) > 0 ]

def get_neighbors(node):
    neighbors = []  
    current_state = node.state  

    for source_stack_index in range(len(current_state)):  # Iterate through all stacks 

        # Moving block to a new stack
        new_state = [list(stack) for stack in current_state]  # Create a deep copy of the current state
        if len(new_state[source_stack_index]) != 1:  # Skip if the source stack has only one block
        
            block = new_state[source_stack_index].pop()  # Pop the top block from the source stack
            new_state.append([block])  # Create a new stack with the block
            neighbors.append((new_state, (block, 'Table')))  # Append the new state and the corresponding move to the neighbors list


        # Moving block to other stacks
        for target_stack_index in range(len(current_state)):  # Iterate through all stacks
            
            if source_stack_index == target_stack_index: continue  # Skip if source and target stacks are the same
                
            new_state = [list(stack) for stack in current_state]  # Create a deep copy of the current state
            
			# Pop the top block from the source stack and move to the target stack
            
            target = new_state[target_stack_index][-1]  
            block = new_state[source_stack_index].pop()  
            new_state[target_stack_index].append(block)  
            new_state = remove_empty_stack(new_state)  # Remove any empty stacks from the new state
             
            neighbors.append((new_state, (block, target)))  # Append the new state and the corresponding move to the neighbors list


    # Convert each neighbor state and move to a Node object before returning
    return [(Node(state, node), move) for state, move in neighbors]

def reconstruct_path(node):
    path = []
    while node.parent:
        path.append(node.move)
        node = node.parent
    return path[::-1]
