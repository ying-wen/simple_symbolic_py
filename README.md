#  A simple module for evaluating scalar formulas

## Requirements
* Python 2.7

## Idea
Construct a symbolic graph, each node in the graph is a 'Block', the node can be a Var, Constant or Operator(+-\*/\*\*), which are Block's subclass. Every Block should implement following methods:

* forward: calculate the result on current node
* backward: calculate the gradient with upstream on current node, and pass the gradient(s) to the args.
* forward_partial: calculate the result excpet specific var.
* deriv: construct the symbolic graph of partial derivative of a var.

## Example
'''
x = Var('x')
y = Var('y')
z = Var('z')
e=x**2+y*z #returns an expression object 
e.eval(x=3, y=4, z=5) # returns 29 (= 3**2 + 4*5) 
e.eval(x=3, y=4) # raises an error
str(e) #returns"x**2+y*z"
'''
