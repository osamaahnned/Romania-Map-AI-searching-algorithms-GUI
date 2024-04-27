import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import *
from PIL import Image, ImageTk
import heapq

############################
# DATA
heuristic = {
        'Arad': 366,
        'Bucharest': 0,
        'Craiova': 160,
        'Drobeta': 242,
        'Eforie': 161,
        'Fagaras': 178,
        'Giurgiu': 77,
        'Hirsova': 151,
        'Iasi': 226,
        'Lugoj': 244,
        'Mehadia': 241,
        'Neamt': 234,
        'Oradea': 380,
        'Pitesti': 98,
        'Rimnicu': 193,
        'Sibiu': 253,
        'Timisoara': 329,
        'Urziceni': 80,
        'Vaslui': 199,
        'Zerind': 374
    }
cost_graph = {
        'Arad': {'Sibiu': 140, 'Zerind': 75, 'Timisoara': 118},
        'Zerind': {'Arad': 75, 'Oradea': 71},
        'Oradea': {'Zerind': 71, 'Sibiu': 151},
        'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu': 80},
        'Timisoara': {'Arad': 118, 'Lugoj': 111},
        'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
        'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
        'Drobeta': {'Mehadia': 75, 'Craiova': 120},
        'Craiova': {'Drobeta': 120, 'Rimnicu': 146, 'Pitesti': 138},
        'Rimnicu': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
        'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
        'Pitesti': {'Rimnicu': 97, 'Craiova': 138, 'Bucharest': 101},
        'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 8},
        'Giurgiu': {'Bucharest': 90},
        'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
        'Hirsova': {'Urziceni': 98, 'Eforie': 86},
        'Eforie': {'Hirsova': 86},
        'Vaslui': {'Iasi': 92, 'Urziceni': 142},
        'Iasi': {'Vaslui': 92, 'Neamt': 87},
        'Neamt': {'Iasi': 87}
    }
graph = {
        'Arad': ['Sibiu', 'Zerind', 'Timisoara'],
        'Zerind': ['Arad', 'Oradea'],
        'Oradea': ['Zerind', 'Sibiu'],
        'Sibiu': ['Arad', 'Oradea', 'Fagaras', 'Rimnicu'],
        'Timisoara': ['Arad', 'Lugoj'],
        'Lugoj': ['Timisoara', 'Mehadia'],
        'Mehadia': ['Lugoj', 'Drobeta'],
        'Drobeta': ['Mehadia', 'Craiova'],
        'Craiova': ['Drobeta', 'Rimnicu', 'Pitesti'],
        'Rimnicu': ['Sibiu', 'Craiova', 'Pitesti'],
        'Fagaras': ['Sibiu', 'Bucharest'],
        'Pitesti': ['Rimnicu', 'Craiova', 'Bucharest'],
        'Bucharest': ['Fagaras', 'Pitesti', 'Giurgiu', 'Urziceni'],
        'Giurgiu': ['Bucharest'],
        'Urziceni': ['Bucharest', 'Vaslui', 'Hirsova'],
        'Hirsova': ['Urziceni', 'Eforie'],
        'Eforie': ['Hirsova'],
        'Vaslui': ['Iasi', 'Urziceni'],
        'Iasi': ['Vaslui', 'Neamt'],
        'Neamt': ['Iasi']
    }
############################


#Breadth First Search BFS
def bfs(graph, start, goal):  #            1

    visited = [] # Explored set
    queue = [(start, [start])] # fronter

    while queue:
        current_node, path = queue.pop(0)

        if current_node == goal:
            return path  # Goal found, return the path

        visited.append(current_node)

        for neighbor in graph[current_node]:
            if neighbor not in visited :
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    return None  # If the goal is not reached

#-------------------------=======================
#Uniform Cost Search UCS
def uniform_cost_search(cost_graph, start, goal):#            2
    # Example usage:

    priority_queue = [(0, start, [])]  # Each element is a tuple (cost, node, path)
    visited = []

    while priority_queue:
        current_cost, current_node, path_so_far = heapq.heappop(priority_queue)

        if current_node == goal:
            return current_cost, path_so_far + [current_node]  # Return cost and path

        if current_node in visited:
            continue

        visited.append(current_node)

        for neighbor, cost in cost_graph[current_node].items():
            if neighbor not in visited:
                total_cost = current_cost + cost
                heapq.heappush(priority_queue, (total_cost, neighbor, path_so_far + [current_node]))

    return None, []  # Return none if no path is found


#--------========================================
#Depth First Search DFS
def dfs_with_goal(graph, start, goal): #            1

    stack = [(start, [start])]
    visited = []

    while stack:
        #print(stack)
        current_node, path = stack.pop()

        if current_node == goal:
            return path

        if current_node in visited:
            continue

        visited.append(current_node)

        for neighbor in reversed(graph[current_node]):
            if neighbor not in visited :
              stack.append((neighbor, path + [neighbor]))

    return None
#----------===============================
#Depth Limited Search DLS
def dls_with_goal(graph, start, goal, depth_limit):#            1

    stack = [(start, [start], 0)]
    visited = []

    while stack:
        current_node, path, depth = stack.pop()
        if current_node == goal:
            return path

        if depth < depth_limit:

            if current_node in visited:
              continue

            visited.append(current_node)
            for neighbor in reversed(graph[current_node]):
              if neighbor not in visited :
                stack.append((neighbor, path + [neighbor], depth + 1))

    return
#---------=======================
#Bidirectional
def bidirectional_search(graph, start, goal):#            1

    start_queue = [(start, [start])]  # Queue for the forward search
    goal_queue = [(goal, [goal])]    # Queue for the backward search

    visited_start = []  # Set to keep track of visited nodes in the forward search
    visited_goal = []  # Set to keep track of visited nodes in the backward search

    # Initialize path lists for forward and backward search
    forward_paths = {start: [start]}
    backward_paths = {goal: [goal]}

    while start_queue and goal_queue:
        # Forward search
        current_start, path_start = start_queue.pop(0)
        visited_start.append(current_start)

        if current_start in visited_goal:
            backward_path = backward_paths[current_start]
            backward_path.pop()
            return forward_paths[current_start] + backward_path[::-1]


        for neighbor in graph[current_start]:
            if neighbor not in visited_start:
                start_queue.append((neighbor, path_start + [neighbor]))
                forward_paths[neighbor] = path_start + [neighbor]

        # Backward search
        current_goal, path_goal = goal_queue.pop(0)
        visited_goal.append(current_goal)

        if current_goal in visited_start:

          forward_path = forward_paths[current_goal]
          forward_path.pop()
          return forward_path + backward_paths[current_goal][::-1]


        for neighbor in graph[current_goal]:
            if neighbor not in visited_goal:
                goal_queue.append((neighbor, path_goal + [neighbor]))
                backward_paths[neighbor] =  path_goal + [neighbor]

    return None  # No path found
#-------==================================
#Depth Limited DFS (Iterative)
def depth_limited_dfs(graph,start, goal, depth_limit):#            1

    stack = [(start, [start], 0)]
    visited = []

    while stack:
        current_node, path, depth = stack.pop()
        if current_node == goal:
            return path

        if depth < depth_limit:

            if current_node in visited:
                continue

            visited.append(current_node)
            for neighbor in reversed(graph[current_node]):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))

    return None
def iterative_deepening_dfs(graph,start, goal):
    depth_limit = 0

    while True:
        path = depth_limited_dfs(graph ,start, goal, depth_limit)

        if path is not None:
            return path
        depth_limit += 1
    return None
#---------==========================
#Greedy Best First Search
def greedy_best_first_search(graph,start, goal,heuristic):#            1

    queue = [(heuristic[start], start, [start])]
    visited = []

    while queue:
        _, node, path = heapq.heappop(queue)

        if node == goal:
            return path  # Return the path from start to goal

        visited.append(node)

        if node in graph:
            neighbors = graph[node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    heapq.heappush(queue, (heuristic[neighbor], neighbor, new_path))

    return None  # No path from start to goal found
#--------==========================
#A*
def a_star_search(cost_graph,start, goal,heuristic):#            2

    queue = [(0 + heuristic[start] , start, [start])]
    visited = []

    while queue:
        cost, node, path = heapq.heappop(queue)
        cost = cost - heuristic[node]

        if node == goal:
            return path, cost  # Return the path and cost from start to goal

        visited.append(node)

        if node in cost_graph:
            neighbors = cost_graph[node]
            for neighbor, edge_cost in neighbors.items():
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    total_cost = new_cost + heuristic[neighbor]
                    heapq.heappush(queue, (total_cost, neighbor, new_path))

    return None  # No path from start to goal found
#-=-=-=-=-=-=-=-=-=-=-=-=-=--

#GUI



root = tk.Tk()
root.title('Romania')
root.geometry("1280x720")

image1 = ImageTk.PhotoImage(Image.open('Simplified-Map-of-Romania.jpg'))
label = Label(root,image=image1)
label.pack()

start_label = tk.Label(text="Start Node:")
start_label.pack()

start_entry = tk.Entry()
start_entry.pack()

goal_label = tk.Label(text="Goal Node:")
goal_label.pack()

goal_entry = tk.Entry()
goal_entry.pack()

result_label = tk.Label(text="Result:")
result_label.pack()

romania_map = ['BFS', 'UCS', 'DFS', 'DLS', 'BS','IDDFS','GBFS','AStar']
string = tk.StringVar(value=romania_map)

listbox = tk.Listbox(
    root,
    listvariable=string,
    height=10,
    selectmode=tk.SINGLE
)
listbox.pack(fill=tk.BOTH)


def items_selected(event):
    selected_indices = listbox.curselection()
    print(selected_indices)
    # get selected items
    selected_langs = ",".join([listbox.get(selected_indices)])
    ch=0
    start_node = start_entry.get()
    goal_node = goal_entry.get()
    if selected_indices[0]==0:
        path_to_goal = bfs(graph, start_node, goal_node)
        ch=0
    elif selected_indices[0]==1:
        cost , path_to_goal = uniform_cost_search(cost_graph,start_node, goal_node)
        ch=1
    elif selected_indices[0]==2:
        path_to_goal = dfs_with_goal(graph,start_node, goal_node)
        ch=0
    elif selected_indices[0]==3:
        depth_limit = 9
        path_to_goal = dls_with_goal(graph,start_node, goal_node, depth_limit)
        ch=0
    elif selected_indices[0]==4:
        path_to_goal = bidirectional_search(graph,start_node, goal_node)
        ch=0
    elif selected_indices[0]==5:
        path_to_goal = iterative_deepening_dfs(graph,start_node, goal_node)
        ch=0
    elif selected_indices[0]==6:
        path_to_goal = greedy_best_first_search(graph,start_node, goal_node,heuristic)
        ch=0
    elif selected_indices[0]==7:
        path_to_goal, cost = a_star_search(cost_graph, start_node, goal_node,heuristic)
        ch=1
    try:
        if path_to_goal:
            if ch:
                result_text = f"Path from {start_node} to {goal_node}: {' -> '.join(path_to_goal)}\nThe cost is: {cost}"
            else:
                result_text = f"Path from {start_node} to {goal_node}: {' -> '.join(path_to_goal)}"
        else:
            result_text = f"No path found from {start_node} to {goal_node}"
    except:
        result_text = "Wrong input"


    # get all selected indices
    # msg_from = f'You selected: {selected_langs_from}'
    # showinfo(title='From', message=msg_from)
    result_label.config(text=result_text)


listbox.bind('<<ListboxSelect>>', items_selected)

root.mainloop()