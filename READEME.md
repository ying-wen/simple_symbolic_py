#  A simple module for evaluating scalar formulas

## Requirements
* Python 2.7

## Idea
Construct a symbolic graph, each node in the graph is a 'Block', the node can be a Var, Constant or Operator(+-\*/\*\*), which are Block's subclass. Every Block should implement following methods:

* forward: calculate the result on current node
* backward: calculate the gradient with upstream on current node, and pass the gradient(s) to the args.
* forward_partial: calculate the result excpet specific var.
* deriv: construct the symbolic graph of partial derivative of a var.

