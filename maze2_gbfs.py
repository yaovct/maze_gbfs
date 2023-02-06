# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:27:39 2023

@author: VictorYang

Add greedy best-first search with Manhattan distance heuristic

1. Override the remove method of queue frontier.
2. Calculate the Manhattan distance from each node to the goal and use it as a priority for the priority queue.
3. In each iteration, pop the node with the lowest cost (i.e. the one closest to the goal) from the priority queue instead of removing from the frontier directly.
4. Also update the basename attribute to include the algorithm used (e.g. self.basename += ".gbfs").

"""
import sys

class Node():
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

class PriorityFrontier(QueueFrontier):
    # Override the remove method only
    def remove(self):
        '''

        Raises
        ------
        empty
            if no element in frontier.

        Returns
        -------
        node
            the node with lowest cost. (frontier must remove this member)

        '''
        if self.empty():
            raise Exception("empty frontier")
        else:
            # Remove the node when its cost is minial
            i = 0
            min_cost = 0
            min_node = 0
            for node in self.frontier:
                if min_cost == 0:
                    min_cost = node.cost
                    min_node = 0
                elif node.cost <= min_cost: # result of '<=' is different from '<'
                    min_cost = node.cost
                    min_node = i
                i += 1
            return self.frontier.pop(min_node)


class Maze():

    def __init__(self, filename):

        import os
        base = os.path.basename(filename)
        self.basename = os.path.splitext(base)[0]
        
        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None


    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("██", end="")
                elif (i, j) == self.start:
                    print("AA", end="")
                elif (i, j) == self.goal:
                    print("BB", end="")
                elif solution is not None and (i, j) in solution:
                    print("**", end="")
                else:
                    #print(" ", end="")
                    # show Manhattan distance to the goal
                    man_dist = abs(i - self.goal[0]) + abs(j - self.goal[1])
                    print(str(man_dist).zfill(2), end="")
            print()
        print()


    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col), abs(row - 1 - self.goal[0]) + abs(col - self.goal[1])),
            ("down", (row + 1, col), abs(row + 1 - self.goal[0]) + abs(col - self.goal[1])),
            ("left", (row, col - 1), abs(row - self.goal[0]) + abs(col - 1 - self.goal[1])),
            ("right", (row, col + 1), abs(row - self.goal[0]) + abs(col + 1 - self.goal[1]))
        ]

        result = []
        for action, (r, c), cost in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c), cost))
        return result


    def solve(self):
        """Finds a solution to maze, if one exists."""

        # Keep track of number of states explored
        self.num_explored = 0

        # Initialize frontier to just the starting position
        start = Node(state=self.start, parent=None, action=None, cost=None)
        frontier = PriorityFrontier()
        self.basename += ".gbfs"
        
        frontier.add(start)

        # Initialize an empty explored set
        self.explored = set()

        # Keep looping until solution found
        while True:

            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception("no solution")

            # Choose a node from the frontier
            node = frontier.remove()
            self.num_explored += 1

            # If node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # Mark node as explored
            self.explored.add(node.state)

            # Add neighbors to frontier
            for action, state, cost in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action, cost=cost)
                    frontier.add(child)


    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze2.py maze.txt")

m = Maze(sys.argv[1])
print("Maze:")
m.print()
print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image(m.basename + ".png", show_explored=True)
