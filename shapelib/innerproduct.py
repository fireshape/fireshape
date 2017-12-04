

class InnerProduct(object):

    def __init__(self):
        pass

    def riesz_map(self, v, out): # dual to primal
        # return A^{-1}*v
        # solve(A, out, v)
        pass

    def eval(self, u, v): # inner product in primal space
        # return u^T A v
        pass

class InterpolatingInnerProduct(InnerProduct):

    def __init__(self, inner_product, interp):
        self.interp = interp
        self.inner_product = inner_product


    def riesz_map(self, v, out):
        # temp = interp.T*v
        # temp2 = interp*temp
        # self.inner_product.riesz_map(temp2, ..)
        # return interpT*...
