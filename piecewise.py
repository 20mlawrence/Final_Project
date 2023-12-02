#Piecewise PREM

import numpy as np

class PiecewisePolynomial(object):
    #create class defining S = sum(c[m, i] * (xp - x[i])**(k-m) for m in (k+1))
    def __init__(self, c, x):
        assert len(x.shape)==1, "1D breakpoints"
        self.breakpoints = x 
        if len(c.shape)==1:
            c = np.expand_dims(c, axis=1)
            c = np.append(c,np.zeros_like(c), axis=1)
        assert len(c.shape)==2, "2D breakpoints"
        self.coeffs = c
        
        
    def __call__(self, xp, break_down=False):
        if np.ndim(xp) == 0:
            value = self._evaluate_at_point(xp, break_down)
        else:
            value = np.zeros_like(xp)
            for i in range(xp.size):
                value[i] = self._evaluate_at_point(xp[i], break_down)
        return value        
        
        
    def _evaluate_at_point(self, x, break_down=False):
        # evaluate at x
        coef = self._get_coefs(x, break_down)
        value = self._evaluate_polynomial(x, coef)
        return value
        
        
    def _evaluate_polynomial(self, x, coef):
        value = 0
        for i, c in enumerate(coef):
            value = value + c * x**i
        return value
        
        
    def _get_coefs(self, x, break_down=False):
        if x == self.breakpoints[-1]:
            return self.coeffs[-1,:]
        if break_down:
            for i in range(self.breakpoints.size):
                if ((x > self.breakpoints[i]) and 
                              (x <= self.breakpoints[i+1])):
                    return self.coeffs[i,:]
        else:
            for i in range(self.breakpoints.size):
                if ((x >= self.breakpoints[i]) and 
                               (x < self.breakpoints[i+1])):
                    return self.coeffs[i,:]
        
        return None
    
    
    def derivative(self):
    
        deriv_breakpoints = self.breakpoints
        deriv_coeffs = np.zeros((self.coeffs.shape[0],
                                 self.coeffs.shape[1]-1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                if i == 0:
                    continue 
                deriv_coeffs[seg,i-1] = self.coeffs[seg,i]*i
                
        deriv = PiecewisePolynomial(deriv_coeffs, deriv_breakpoints)
        return deriv
    
    def antiderivative(self):
        
        antideriv_breakpoints = self.breakpoints
        antideriv_coeffs = np.zeros((self.coeffs.shape[0],
                                     self.coeffs.shape[1]+1))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                antideriv_coeffs[seg,i+1] = self.coeffs[seg,i]/(i+1)
                
        antideriv = PiecewisePolynomial(antideriv_coeffs, 
                                        antideriv_breakpoints)
        return antideriv
    
    
    def integrate(self, a, b):
        
        integral = 0
        lower_bound = a
        for bpi, bp in enumerate(self.breakpoints):
            if bp > lower_bound:
                if self.breakpoints[bpi] >= b:
                    integral = integral + (self(b, break_down=True) - 
                                           self(lower_bound))
                    break
                else:
                    integral = integral + (self(bp, break_down=True) - 
                                           self(lower_bound))
                    lower_bound = bp

        return integral
    
    
    def mult(self, other):
        assert self.coeffs.shape[0] == other.coeffs.shape[0], \
                                     'different number of breakpoints'
        mult_breakpoints = self.breakpoints
        mult_coefs = np.zeros((self.coeffs.shape[0],
                               self.coeffs.shape[1]+other.coeffs.shape[1]))
        for seg in range(self.coeffs.shape[0]):
            for i in range(self.coeffs.shape[1]):
                for j in range(other.coeffs.shape[1]):
                    mult_coefs[seg,i+j] = mult_coefs[seg,i+j] + \
                                 self.coeffs[seg,i] * other.coeffs[seg,j]
                    
        mult_poly = PiecewisePolynomial(mult_coefs, mult_breakpoints)
        return mult_poly
