from scalar import Var

x = Var('x')
y = Var('y')
z = Var('z')
f = (x ** 2) + y * z
print 'Formula of f: ' + str(f)
print 'Result of f: ' + str(f.eval(x=3, y=4, z=5))
dervix = f.deriv('x')
print 'Formula of df/dx: ' + str(dervix)
print 'Gradient of x: ' + str(dervix.eval(x=3))

print '\nThe gradient of x also can be directly calculated by backward method'
f.forward()  # make sure the forware is used becore backward
f.backward()
print 'Gradient of x: ' + str(x.gradParam)

print '\n-----------More Complex Case--------------\n'

m = Var('m')
n = Var('n')
h = Var('h')
y = -(m ** 2) * n + h / m + m * n * h
print 'Formula of y: ' + str(y)
print 'Result of y: ' + str(y.eval(m=3, n=4, h=5))
dervix = y.deriv('m')
print 'Formula of dy/dm: ' + str(dervix)
print 'Gradient of x: ' + str(dervix.eval(m=3))
