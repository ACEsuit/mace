import torch
from functools import reduce
from torch.optim.optimizer import Optimizer

import math

be_verbose=False

class LBFGSNew(Optimizer):
    """Implements L-BFGS algorithm.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (fallback value when line search fails. not really needed) (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 10)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 7).
        line_search_fn: if True, use cubic interpolation to findstep size, if False: fixed step size (default: True)
        batch_mode: True for stochastic version (default: False)
        cost_use_gradient: set this to True when the cost function also needs the gradient, for example in TV (total variation) regularization. (default: False)

        Example usage for full batch mode:

          optimizer = LBFGSNew(model.parameters(), history_size=7, max_iter=100, line_search_fn=True, batch_mode=False)

        Example usage for batch mode (stochastic):

          optimizer = LBFGSNew(net.parameters(), history_size=7, max_iter=4, line_search_fn=True,batch_mode=True)
          Note: when using a closure(), only do backward() after checking the gradient is available,
          Eg: 
            def closure():
             optimizer.zero_grad()
             outputs=net(inputs)
             loss=criterion(outputs,labels)
             if loss.requires_grad:
               loss.backward()
             return loss

        Note: Some cost functions also use the gradient itself (for example as a regularization term). In this case, you need to set cost_use_gradient=True.

    """

    def __init__(self, params, lr=1, max_iter=10, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=7,
                 line_search_fn=True, batch_mode=False, cost_use_gradient=False):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn,
                        batch_mode=batch_mode, cost_use_gradient=cost_use_gradient)
        super(LBFGSNew, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().contiguous().view(-1)
            else:
                view = p.grad.data.contiguous().view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(update[offset:offset + numel].view_as(p.data), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    #FF copy the parameter values out, create a single vector
    def _copy_params_out(self):
        return [p.detach().clone(memory_format=torch.contiguous_format) for p in self._params]

    #FF copy the parameter values back, dividing the vector into a list
    def _copy_params_in(self,new_params):
        with torch.no_grad():
          for p, pdata in zip(self._params, new_params):
            p.copy_(pdata)

    #FF line search xk=self._params, pk=step direction, gk=gradient, alphabar=max. step size
    def _linesearch_backtrack(self,closure,pk,gk,alphabar):
        """Line search (backtracking)

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            pk: step direction vector
            gk: gradient vector 
            alphabar: max step size
        """


        # constants (FIXME) find proper values
        # c1: large values better for small batch sizes
        c1=1e-4
        citer=35
        alphak=alphabar# default return step
 
        # state parameter 
        state = self.state[self._params[0]]

        # make a copy of original params
        xk=self._copy_params_out()

   
        f_old=float(closure())
        # param = param + alphak * pk
        self._add_grad(alphak, pk)
        f_new=float(closure())

        # prod = c1 * ( alphak ) * gk^T pk = alphak * prodterm
        s=gk
        prodterm=c1*(s.dot(pk))

        ci=0
        if be_verbose:
         print('LN %d alpha=%f fnew=%f fold=%f prod=%f'%(ci,alphak,f_new,f_old,prodterm))
        # catch cases where f_new is NaN
        while (ci<citer and (math.isnan(f_new) or  f_new > f_old + alphak*prodterm)):
           alphak=0.5*alphak
           self._copy_params_in(xk)
           self._add_grad(alphak, pk)
           f_new=float(closure())
           if be_verbose:
             print('LN %d alpha=%f fnew=%f fold=%f'%(ci,alphak,f_new,f_old))
           ci=ci+1

        # if the cost is not sufficiently decreased, also try -ve steps
        if (f_old-f_new < torch.abs(prodterm)):
          alphak1=-alphabar
          self._copy_params_in(xk)
          self._add_grad(alphak1, pk)
          f_new1=float(closure())
          if be_verbose:
            print('NLN fnew=%f'%f_new1)
          while (ci<citer and (math.isnan(f_new1) or  f_new1 > f_old + alphak1*prodterm)):
             alphak1=0.5*alphak1
             self._copy_params_in(xk)
             self._add_grad(alphak1, pk)
             f_new1=float(closure())
             if be_verbose:
               print('NLN %d alpha=%f fnew=%f fold=%f'%(ci,alphak1,f_new1,f_old))
             ci=ci+1

          if f_new1<f_new:
            # select -ve step
            alphak=alphak1

        # recover original params
        self._copy_params_in(xk)
        # update state
        state['func_evals'] += ci
        return alphak



    #FF line search xk=self._params, pk=gradient
    def _linesearch_cubic(self,closure,pk,step):
        """Line search (strong-Wolfe)

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            pk: gradient vector 
            step: step size for differencing 
        """

        # constants
        alpha1=10*self.param_groups[0]['lr']#10.0
        sigma=0.1
        rho=0.01
        t1=9 
        t2=0.1
        t3=0.5
        alphak=self.param_groups[0]['lr']# default return step
 
        # state parameter 
        state = self.state[self._params[0]]

        # make a copy of original params
        xk=self._copy_params_out()
   
        phi_0=float(closure())
        tol=min(phi_0*0.01,1e-6)

        # xp <- xk+step. pk
        self._add_grad(step, pk) #FF param = param + t * grad 
        p01=float(closure())
        # xp <- xk-step. pk
        self._add_grad(-2.0*step, pk) #FF param = param - t * grad 
        p02=float(closure())

        ##print("p01="+str(p01)+" p02="+str(p02))
        gphi_0=(p01-p02)/(2.0*step)
        ##print("tol="+str(tol)+" phi_0="+str(phi_0)+" gphi_0="+str(gphi_0))
        # catch instances when step size is too small 
        if abs(gphi_0)<1e-12:
          return 1.0

        mu=(tol-phi_0)/(rho*gphi_0)
        # catch if mu is not finite
        if math.isnan(mu):
           return 1.0

        ##print("mu="+str(mu))
        
        # counting function evals
        closure_evals=3

        ci=1
        alphai=alpha1 # initial value for alpha(i) : check if 0<alphai<=mu 
        alphai1=0.0
        phi_alphai1=phi_0
        while (ci<4) : # FIXME
          # evalualte phi(alpha(i))=f(xk+alphai pk)
          self._copy_params_in(xk) # original
          # xp <- xk+alphai. pk
          self._add_grad(alphai, pk) #
          phi_alphai=float(closure())
          if phi_alphai<tol:
             alphak=alphai 
             if be_verbose:
              print("Linesearch: condition 0 met")
             break
          if (phi_alphai>phi_0+alphai*gphi_0) or (ci>1 and phi_alphai>=phi_alphai1) :
             # ai=alphai1, bi=alphai bracket
             if be_verbose:
              print("bracket "+str(alphai1)+","+str(alphai))
             alphak=self._linesearch_zoom(closure,xk,pk,alphai1,alphai,phi_0,gphi_0,sigma,rho,t1,t2,t3,step)
             if be_verbose:
              print("Linesearch: condition 1 met") 
             break

          # evaluate grad(phi(alpha(i))) */
          # note that self._params already is xk+alphai. pk, so only add the missing term
          # xp <- xk+(alphai+step). pk
          self._add_grad(step, pk) #FF param = param - t * grad 
          p01=float(closure())
          # xp <- xk+(alphai-step). pk
          self._add_grad(-2.0*step, pk) #FF param = param - t * grad 
          p02=float(closure())
          gphi_i=(p01-p02)/(2.0*step);
        
          if (abs(gphi_i)<=-sigma*gphi_0):
             alphak=alphai
             if be_verbose:
              print("Linesearch: condition 2 met") 
             break

          if gphi_i>=0.0 :
             # ai=alphai, bi=alphai1 bracket
             if be_verbose:
              print("bracket "+str(alphai)+","+str(alphai1))
             alphak=self._linesearch_zoom(closure,xk,pk,alphai,alphai1,phi_0,gphi_0,sigma,rho,t1,t2,t3,step)
             if be_verbose:
              print("Linesearch: condition 3 met") 
             break
          # else preserve old values
          if (mu<=2.0*alphai-alphai1):
             alphai1=alphai
             alphai=mu
          else:
             # choose by interpolation in [2*alphai-alphai1,min(mu,alphai+t1*(alphai-alphai1)] 
            p01=2.0*alphai-alphai1;
            p02=min(mu,alphai+t1*(alphai-alphai1))
            alphai=self._cubic_interpolate(closure,xk,pk,p01,p02,step)


          phi_alphai1=phi_alphai;
          # update function evals
          closure_evals +=3
          ci=ci+1

          


        # recover original params
        self._copy_params_in(xk)
        # update state
        state['func_evals'] += closure_evals
        return alphak


    def _cubic_interpolate(self,closure,xk,pk,a,b,step):
        """ Cubic interpolation within interval [a,b] or [b,a] (a>b is possible)
          
           Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            xk: copy of parameter values 
            pk: gradient vector 
            a/b:  interval for interpolation
            step: step size for differencing 
        """


        self._copy_params_in(xk)

        # state parameter 
        state = self.state[self._params[0]]
        # count function evals
        closure_evals=0

        # xp <- xk+a. pk
        self._add_grad(a, pk) #FF param = param + t * grad 
        f0=float(closure())
        # xp <- xk+(a+step). pk
        self._add_grad(step, pk) #FF param = param + t * grad 
        p01=float(closure())
        # xp <- xk+(a-step). pk
        self._add_grad(-2.0*step, pk) #FF param = param - t * grad 
        p02=float(closure())
        f0d=(p01-p02)/(2.0*step)

        # xp <- xk+b. pk
        self._add_grad(-a+step+b, pk) #FF param = param + t * grad 
        f1=float(closure())
        # xp <- xk+(b+step). pk
        self._add_grad(step, pk) #FF param = param + t * grad 
        p01=float(closure())
        # xp <- xk+(b-step). pk
        self._add_grad(-2.0*step, pk) #FF param = param - t * grad 
        p02=float(closure())
        f1d=(p01-p02)/(2.0*step)

        closure_evals=6

        aa=3.0*(f0-f1)/(b-a)+f1d-f0d
        p01=aa*aa-f0d*f1d
        if (p01>0.0):
           cc=math.sqrt(p01)
           #print('f0='+str(f0d)+' f1='+str(f1d)+' cc='+str(cc))
           if (f1d-f0d+2.0*cc)==0.0:
             return (a+b)*0.5
           z0=b-(f1d+cc-aa)*(b-a)/(f1d-f0d+2.0*cc)
           aa=max(a,b)
           cc=min(a,b)
           if z0>aa or z0<cc:
             fz0=f0+f1
           else:
             # xp <- xk+(a+z0*(b-a))*pk
             self._add_grad(-b+step+a+z0*(b-a), pk) #FF param = param + t * grad 
             fz0=float(closure())
             closure_evals +=1

           # update state
           state['func_evals'] += closure_evals

           if f0<f1 and f0<fz0:
             return a

           if f1<fz0:
             return b
           # else
           return z0
        else:

           # update state
           state['func_evals'] += closure_evals

           if f0<f1:
             return a
           else:
             return b

        # update state
        state['func_evals'] += closure_evals

        # fallback value
        return (a+b)*0.5
     



    #FF bracket [a,b]
    # xk: copy of parameters, use it to refresh self._param 
    def _linesearch_zoom(self,closure,xk,pk,a,b,phi_0,gphi_0,sigma,rho,t1,t2,t3,step):
        """Zoom step in line search

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
            xk: copy of parameter values 
            pk: gradient vector 
            a/b:  bracket interval for line search, 
            phi_0: phi(0)
            gphi_0: grad(phi(0))
            sigma,rho,t1,t2,t3: line search parameters (from Fletcher) 
            step: step size for differencing 
        """

        # state parameter 
        state = self.state[self._params[0]]
        # count function evals
        closure_evals=0

        aj=a
        bj=b
        ci=0
        found_step=False
        while ci<4: # FIXME original 10
           # choose alphaj from [a+t2(b-a),b-t3(b-a)]
           p01=aj+t2*(bj-aj)
           p02=bj-t3*(bj-aj)
           alphaj=self._cubic_interpolate(closure,xk,pk,p01,p02,step)

           # evaluate phi(alphaj)
           self._copy_params_in(xk)
           # xp <- xk+alphaj. pk
           self._add_grad(alphaj, pk) #FF param = param + t * grad 
           phi_j=float(closure())
          
           # evaluate phi(aj)
           # xp <- xk+aj. pk
           self._add_grad(-alphaj+aj, pk) #FF param = param + t * grad 
           phi_aj=float(closure())

           closure_evals +=2

           if (phi_j>phi_0+rho*alphaj*gphi_0) or phi_j>=phi_aj :
              bj=alphaj # aj is unchanged
           else:
              # evaluate grad(alphaj)
              # xp <- xk+(alphaj+step). pk
              self._add_grad(-aj+alphaj+step, pk) #FF param = param + t * grad 
              p01=float(closure())
              # xp <- xk+(alphaj-step). pk
              self._add_grad(-2.0*step, pk) #FF param = param + t * grad 
              p02=float(closure())
              gphi_j=(p01-p02)/(2.0*step)
        

              closure_evals +=2

              # termination due to roundoff/other errors pp. 38, Fletcher
              if (aj-alphaj)*gphi_j <= step:
                 alphak=alphaj
                 found_step=True
                 break
             
              if abs(gphi_j)<=-sigma*gphi_0 :
                 alphak=alphaj
                 found_step=True
                 break

              if gphi_j*(bj-aj)>=0.0:
                 bj=aj
              # else bj is unchanged
              aj=alphaj


           ci=ci+1
        
        if not found_step:
          alphak=alphaj

        # update state
        state['func_evals'] += closure_evals

        return alphak


    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        batch_mode = group['batch_mode']
        cost_use_gradient = group['cost_use_gradient']


        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)


        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        abs_grad_sum = flat_grad.abs().sum()

        if torch.isnan(abs_grad_sum) or abs_grad_sum <= tolerance_grad:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0

        if batch_mode:
          alphabar=lr
          lm0=1e-6

        # optimize for a max of max_iter iterations
        grad_nrm=flat_grad.norm().item()
        while n_iter < max_iter and not math.isnan(grad_nrm):
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                H_diag = 1
                if batch_mode:
                 running_avg=torch.zeros_like(flat_grad.data)
                 running_avg_sq=torch.zeros_like(flat_grad.data)
            else:
                if batch_mode:
                 running_avg=state.get('running_avg')
                 running_avg_sq=state.get('running_avg_sq')
                 if running_avg is None:
                  running_avg=torch.zeros_like(flat_grad.data)
                  running_avg_sq=torch.zeros_like(flat_grad.data)

                # do lbfgs update (update memory) 
                # what happens if current and prev grad are equal, ||y||->0 ??
                y = flat_grad.sub(prev_flat_grad)

                s = d.mul(t)

                if batch_mode: # y = y+ lm0 * s, to have a trust region
                  y.add_(s,alpha=lm0)

                ys = y.dot(s)  # y^T*s
                sn = s.norm().item()  # ||s||
                # FIXME batch_changed does not work for full batch mode (data might be the same)
                batch_changed= batch_mode and (n_iter==1 and state['n_iter']>1)
                if batch_changed: # batch has changed
                   # online estimate of mean,variance of gradient (inter-batch, not intra-batch)
                   # newmean <- oldmean + (grad - oldmean)/niter
                   # moment <- oldmoment + (grad-oldmean)(grad-newmean)
                   # variance = moment/(niter-1)

                   g_old=flat_grad.clone(memory_format=torch.contiguous_format)
                   g_old.add_(running_avg,alpha=-1.0) # grad-oldmean
                   running_avg.add_(g_old,alpha=1.0/state['n_iter']) # newmean
                   g_new=flat_grad.clone(memory_format=torch.contiguous_format)
                   g_new.add_(running_avg,alpha=-1.0) # grad-newmean
                   running_avg_sq.addcmul_(g_new,g_old,value=1) # +(grad-newmean)(grad-oldmean)
                   alphabar=1/(1+running_avg_sq.sum()/((state['n_iter']-1)*(grad_nrm)))
                   if be_verbose:
                     print('iter %d |mean| %f |var| %f ||grad|| %f step %f y^Ts %f alphabar=%f'%(state['n_iter'],running_avg.sum(),running_avg_sq.sum()/(state['n_iter']-1),grad_nrm,t,ys,alphabar))


                if ys > 1e-10*sn*sn and not batch_changed :
                    # updating memory (only when we have y within a single batch)
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                if math.isnan(H_diag):
                  print('Warning H_diag nan')

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'ro' not in state:
                    state['ro'] = [None] * history_size
                    state['al'] = [None] * history_size
                ro = state['ro']
                al = state['al']

                for i in range(num_old):
                    ro[i] = 1. / old_dirs[i].dot(old_stps[i])

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i],alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i],alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)

            else:
                prev_flat_grad.copy_(flat_grad)

            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / abs_grad_sum) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            if math.isnan(gtd.item()):
              print('Warning grad norm infinite')
              print('iter %d'%state['n_iter'])
              print('||grad||=%f'%grad_nrm)
              print('||d||=%f'%d.norm().item())
            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn:
                # perform line search, using user function
                ##raise RuntimeError("line search function is not supported yet")
                #FF#################################
                # Note: we disable gradient calculation during line search
                # because it is not needed
                if not cost_use_gradient:
                 torch.set_grad_enabled(False)
                if not batch_mode:
                 t=self._linesearch_cubic(closure,d,1e-6) 
                else:
                 t=self._linesearch_backtrack(closure,d,flat_grad,alphabar)
                if not cost_use_gradient:
                 torch.set_grad_enabled(True)

                if math.isnan(t):
                  print('Warning: stepsize nan')
                  t=lr
                self._add_grad(t, d) #FF param = param + t * d 
                if be_verbose:
                 print('step size=%f'%(t))
                #FF#################################
            else:
                #FF Here, t = stepsize,  d = -grad, in cache
                # no line search, simply move with fixed-step
                self._add_grad(t, d) #FF param = param + t * d 
            if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    abs_grad_sum = flat_grad.abs().sum()
                    if math.isnan(abs_grad_sum):
                       print('Warning: gradient nan')
                       break
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            if abs_grad_sum <= tolerance_grad:
                break

            if gtd > -tolerance_change:
                break

            if d.mul(t).abs_().sum() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        if batch_mode:
         if 'running_avg' not in locals() or running_avg is None:
           running_avg=torch.zeros_like(flat_grad.data)
           running_avg_sq=torch.zeros_like(flat_grad.data)
         state['running_avg']=running_avg
         state['running_avg_sq']=running_avg_sq
   

        return orig_loss
