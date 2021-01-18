import numpy as np
import random

class data_stct:
    def __init__(self,args):
        self.args = args
        
    def factors(self):
        high_ = lambda x,_:(max(x[:,1])-x[0][0])/self.args.p_unit
        low_ = lambda x,_:(x[0][0]-min(x[:,2]))/self.args.p_unit
        delta_ = lambda x,_:(x[-1][3]-x[0][0])/self.args.p_unit
        p1_ = lambda x,_:(low_(x,None)+delta_(x,None))/(high_(x,None) +low_(x,None)) if low_(x,None)+high_(x,None)!=0 else random.random() 
        p2_ = lambda x,_:(high_(x,None)-delta_(x,None))/(high_(x,None) +low_(x,None)) if low_(x,None)+high_(x,None)!=0 else random.random()
        volume_ = lambda x,_:sum(x[:,4])/self.args.v_unit
        if self.args.is_tick:
            buy_sell_ = lambda x,_: max(x[-1][5],0.5)
            buy1_ = lambda x,_:x[-1][6]/self.args.o_unit
            sell1_ = lambda x,_:x[-1][7]/self.args.o_unit
            return [high_,low_,delta_,p1_,p2_,volume_,buy_sell_ ,buy1_,sell1_]
        #acc_ = lambda x,x0:(delta_(x,None)-x0[2])*0.9+0.1*x0[5] if x0 is not None else 0
        return [high_,low_,delta_,p1_,p2_,volume_]
    

    def fct_use(self,x):
        sigmoid = lambda x:1/(1+ np.exp(-x/5))        
        while True:
            x,y = [],x
            data = yield y
            for f in self.factors()[:3]:
                x.append(sigmoid(f(data,y)))
            for f in self.factors()[3:]:
                x.append(f(data,y))
                
                
    def resample(self,data):
        #'open','high','low','close','volume','up_down','buy1','sell1'
        t0,v = data[-1,0],data[-1,2]
        l = [[data[-1,1],data[-1,1],data[-1,1],data[-1,1],0,data[-1,1]-data[-1,3],data[-1,4],data[-1,6]]]
        for x in reversed(data[:-1]):
            if t0-x[0] >= self.args.bar_s:
                l[-1][4],v,t0 =v-x[2],x[2],x[0]
                l.append([x[1],x[1],x[1],x[1],0,x[1]-x[3],x[4],x[6]])
            else:
                l[-1][0],l[-1][1],l[-1][2] = x[1],max(l[-1][1],x[1]),min(l[-1][2],x[1])
        return list(reversed(l[:-1]))                
                
                
    def pct(self,data):
        data = np.array(data)
        if self.args.is_tick:
            data = np.array(self.resample(data)[-self.args.in_seq_length-self.args.out_seq_length:])
        empty = np.zeros((self.args.out_seq_length,data.shape[-1] ))
        data = np.concatenate((data,empty),axis =0)
        f = self.fct_use(None)
        next(f)
        x = [f.send(np.array([d])) for d in data[:self.args.in_seq_length]]
        y = [[f.send(data[i-c+1:i+1]) for i in range(self.args.in_seq_length+c-1,self.args.in_seq_length+self.args.out_seq_length,c)] for c in self.args.conv]
        return [x,y]