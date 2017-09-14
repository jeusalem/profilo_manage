
ERRO = 1e-7
class Equation(object):

    #b_i and b_imp don`t include the ratio of cash shape=[num_stocks]
    def __init__(self, b_imp, b_i, c_s, c_p):

        self.b_imp = b_imp
        self.c_s = c_s
        self.c_p = c_p
        self.b_i = b_i
        self.erro = ERRO

    def sumPlus(self, ut):
        # print('b________imp',b_imp)
        size = len(self.b_imp)
        sum_bimp_utbi = 0
        sum_utbi_bimp = 0
        # print('b_imp',self.b_imp)
        # print('b_i',self.b_i)
        tmp = 0
        # print('size',size)
        for i in range(size):
            tmp = self.b_imp[i] - ut * self.b_i[i]
            # print(tmp,i)
            if tmp >0:
                sum_bimp_utbi += tmp
            else:
                sum_utbi_bimp += tmp * (-1)    # always keep postive number
        return (sum_bimp_utbi, sum_utbi_bimp)


    def get_u_t(self):
        ut1 = 0.5
        while(True):
            (s, p) = self.sumPlus(ut1)
            # print(s,p)
            ut2 = 1 - self.c_s * s - self.c_p * p
            # print('cs',self.c_s)
            # print('cp',self.c_p)
            # print('ut2',ut2)
            if abs(ut2-ut1) < self.erro:
                return ut2
            ut1 = ut2




c_s = 0.003
c_p = 0.002

b_imp = [0.2,0.3,0.1,0.2]
b_i = [0.2,0.2,0.2,0.15]
pc = 1
pa = 0
loss = 1e-7


if __name__ == '__main__':
    e = Equation(b_imp,b_i,c_s,c_p)
    ut = e.get_u_t()
    print(ut)