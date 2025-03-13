---
title: State-Space Search
layout: collection
permalink: /Foundations-of-Artificial-Intelligence/State-Space-Search
collection: Foundations-of-Artificial-Intelligence
entries_layout: grid
mathjax: true
toc: true
categories:
  - study
tags:
  - programming
---

# State Spaces

Here consider our state space to

- Static
- Determenistic
- Fully observable
- Discrete
- Single-agent
- Problem-specific

## Formalization

A state space or transition system is a 6-tuple $\mathcal{S} = \langle S, A, cost, T, s_I, S_G \rangle$ with
- Finite set of states $S$
- Finite set of actions $A$
- Action costs: $cost: A \rightarrow \mathbb{R}_0^+$
- Transition relation $T \subseteq S \times A \times S$
- Initial state $s_I \in S$
- Goal states $S_G \subseteq S$

Consider a state space as defined above.
The triples $\langle s, a, s' \rangle \in T$ are called (state) transitions.
$S$ has a transition $\langle s, a, s'\rangle$ if $\langle s, a, s'\rangle \in T$. We also write this as $s \xrightarrow[]{a} s'$.
Transitions are determenistic in $\langle s, a\rangle$: it is forbidden to have both $s \xrightarrow[]{a} s_1$ and $s \xrightarrow[]{a} s_2$. with $s_1 \neq s_2$.

Let $s,s' \in S$ be states with $s \rightarrow s'$:
- $s$ is a predecessor of $s'$
- $s'$ is a successor of $s$
If $s \xrightarrow[]{a} s'$, then action $a$ is applicable in $s$.

Let $s_0,...,s_n\in S$ be states and $a_1,...,a_n \in A$ be actions such that $s_0 \xrightarrow[]{a_1} s_1,...,s_{n-1} \xrightarrow[]{a_n} s_n$
- $\pi = \langle a_1,...,a_n \rangle$ is a path from $s_0$ to $s_1$.
- length of $\pi$: $|\pi| = n$
- cost of $\pi$: $cost(\pi) = \sum_{i=1}^{n} cost(a_i)$

- State $s$ is reachable if a path from $s_I$ to $s$ exists
- Paths from $s \in S$ to some state $s_G \in S_G$ are solutions for/from $s$
- Solutions for $s_I$ are called solutions for $S$
- Optimal solutions (for $s$) have minimal costs among all solutions (for $s$)

## Explicit Graphs

In explicit graphs we represent the state space as a directed graph. The vertices are states and the directed arcs are transitions. This form can be represented using an adjacency list / matrix.

This method is often impossable for large state spaces, as it requires to much memory. If the space is small enough, solutions are easy to compute using for example the Dijkstra's algorithm.

## Declerative Representation

Here the state space is represented declaratively. It provides a compact description of state space as input to algorithms, where then the algorithm directly operates on the compact description.

## Black Box

Here the state space can be seen as an abstract interface with methods.

- init() : Generate initial state
- is_goal(s) : Check if state is goal
- succ(s) : Generate applicable actions and successors of $s$
- cost(a) : Gives cost of action $a$

## Search Algorithms

In a search algorithm we iteratively create a search tree.
We start with the initial state and from there expand a state by generating its successors. This process is repeated when the goal state is expanded / generated, or all reachable states have been considered.

For the search algorithm we consider three data structures
- Search Node: Stores a state that has been reached, how it was reached, and at which cost.
- Open List: Efficiently organizes leaves of search tree
- Closed List: Remembers expanded states to avoid duplicated expansion of the same state

### Search Node

A search node stores a state that has been reached, how it was reached and at which cost. They form the search tree.

- state: State associated with node
- parent: Search node that generated node
- action: action leading from parent to node
- path_cost: cost of path from $s_I$ to node ($g(n)$)

### Open List

The open list organizes the leaves of a search tree. It must support two operations efficiently
- Determine and remove the next node to expand
- Insert a new node that is a candidate node for expansion

- is_empty() : Check if open list is empty
- pop() : Remove and return the next node to expand
- insert(n) : Insert node $n$ into the open list

The open list determines the strategy for which node to expand next.

A efficient implementation is a heap.

### Closed List

The closed list remembers expanded states to avoid duplicated expansion of the same state.

It must support two operations efficiently
- Insert a node whose state is not yet in the closed list
- Test if a node with a given state is in the closed list; if yes, return it

- insert(n) : Inser node $n$ into closed
- lookup(s) : Test if a node with state s exists in the closed list. If yes return it; otherwise return none

A efficient implementation is a hash table.

## Search types

### Tree search

In a tree search we organize the possible paths to be explored in a tree, the search tree. The search nodes then correspond 1:1 to paths from the initial state. Duplicates are allowed and we do not hold a closed list. Therefor it can also have an unbounded depth.

```python
open := new OpenList
open.insert(make_root_node())
# Iterate over not expanded nodes
while not open.is_empty():
    # Expand node
    n := open.pop()
    # If goal get path from start to goal
    if is_goal(n.state):
        return extract_path(n)
    # Iterate over successor and add them to open list for later expansion
    for each (a, s') in succ(n.state):
        n' := make_node(n, a, s')
        open.insert(n')
return unsolvable
```

### Graph search

In a graph search we recognize duplicates and don't expand them again. The search nodes correspond then 1:1 to reachable states. Because of this the depth of a search tree is bounded.

```python
open := new OpenList
open.insert(make_root_node())
closed := new ClosedList
# Iterate over not expanded nodes
while not open.is_empty():
    # Expand node
    n := open.pop()
    if closed.lookup(n.state) = none:
        closed.insert(n)
        # If goal get path from start to goal
        if is_goal(n.state):
            return extract_path(n)
        # Iterate over successor and add them to open list for later expansion
        for each (a, s') in succ(n.state):
            n' := make_node(n, a, s')
            open.insert(n')
return unsolvable
```


## Evaluating Search Algorithms

### Completeness

A search algorithm is semi complete if
- Is guaranteed to find a solution if one exists

A search algorithm is complete if
- It is semi complete
- Terminates if no solution exists

### Optimality

A search algorithm is optimal if
- The solution returned by the search algorithm always has the optimal cost

### Time Complexity

Time complexity is how much time the algorithm needs until termination for the worst case. Here it is usually measured the amount of generated nodes.
It is a function of the branching factor (max number of successors of a state) and the search depth (longest path in the generated search tree).

### Space Complexity

Space Complexity is how much memory the algorithm uses for the worst case. Here it is usually measured in the amount of concurrently stored nodes.
It is a fucntion of the branching factor and the search depth.

## Uniform Cost Search

In the uniform cost search, in comparison to BFS, we expand the node with the minimal path cost ($g(n)$).

```python
open := new MinHeap #Ordered by g
open.insert(make_root_node())
closed := new HashSet
while not open.is_empty():
    # Get node with minimal path
    n := open.pop_min()
    # If not yet explored 
    if n.state not in closed:
        closed.insert(n.state)
        if is_goal(n.state):
            return extract_path(n)
        for each (a, s') in succ(n.state):
            n' := make_node(n, a, s')
            open.insert(n')
return unsolvable
```

Here early goal check is not done because there may be a better path then the first occurence of the goal in the successors. A tree variant is also often not used as it is not even semi complete, as it may run in loops. It is identical to the Dijkstra's algorithm for the shortest paths.

- Complete
- Optimal
- Time- and Space Complexity : $O(b^{\lfloor c^* / \epsilon \rfloor + 1})$, $\epsilon := \min_{a \in A} cost(a) > 0$, $c^*$ optimal solution cost and $b \geq 2$ the branching factor 



```python

```
