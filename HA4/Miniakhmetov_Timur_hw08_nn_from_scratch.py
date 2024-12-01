#!/usr/bin/env python
# coding: utf-8

# Credits: this notebook belongs to [Practical DL](https://docs.google.com/forms/d/e/1FAIpQLScvrVtuwrHSlxWqHnLt1V-_7h2eON_mlRR6MUb3xEe5x9LuoA/viewform?usp=sf_link) course by Yandex School of Data Analysis.

# In[2]:


import numpy as np


# **Module** is an abstract class which defines fundamental methods necessary for a training a neural network. You do not need to change anything here, just read the comments.

# In[3]:


class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        gradInput = module.backward(input, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self,input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput
    

    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.
        
        Make sure to both store the data in `output` field and return it. 
        """
        
        # The easiest case:
            
        # self.output = input 
        # return self.output
        
        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input. 
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.
        
        The shape of `gradInput` is always the same as the shape of `input`.
        
        Make sure to both store the gradients in `gradInput` field and return it.
        """
        
        # The easiest case:
        
        # self.gradInput = gradOutput 
        # return self.gradInput
        
        pass   
    
    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zeroGradParameters(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def getParameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


# # Sequential container

# **Define** a forward and backward pass procedures.

# In[4]:


class Sequential(Module):
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add(self, module):
        self.modules.append(module)

    def updateOutput(self, input):
        self.output = input
        for module in self.modules:
            self.output = module.forward(self.output)
        return self.output

    def backward(self, input, gradOutput):
        # Сохраняем промежуточные выходы
        outputs = [input]
        curr_input = input
        for module in self.modules:
            curr_input = module.forward(curr_input)
            outputs.append(curr_input)
            
        # Обратный проход
        curr_grad = gradOutput
        for i in reversed(range(len(self.modules))):
            curr_grad = self.modules[i].backward(outputs[i], curr_grad)
            
        self.gradInput = curr_grad
        return self.gradInput
      

    def zeroGradParameters(self): 
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]
    
    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self,x):
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


# # Layers

# ## 1. Linear transform layer
# Also known as dense layer, fully-connected layer, FC-layer, InnerProductLayer (in caffe), affine transform
# - input:   **`batch_size x n_feats1`**
# - output: **`batch_size x n_feats2`**

# In[5]:


class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        self.output = np.dot(input, self.W.T) + self.b
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradW = np.dot(gradOutput.T, input)
        self.gradb = np.sum(gradOutput, axis=0)
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q


# ## 2. SoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# $\text{softmax}(x)_i = \frac{\exp x_i} {\sum_j \exp x_j}$
# 
# Recall that $\text{softmax}(x) == \text{softmax}(x - \text{const})$. It makes possible to avoid computing exp() from large argument.

# In[6]:


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # Вычитаем максимальное значение для численной стабильности
        shifted_input = input - np.max(input, axis=1, keepdims=True)
        exp_input = np.exp(shifted_input)
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.zeros_like(input)
        for i in range(len(input)):
            jacobian = np.diag(self.output[i]) -                       np.outer(self.output[i], self.output[i])
            self.gradInput[i] = np.dot(jacobian, gradOutput[i])
        return self.gradInput
    
    def __repr__(self):
        return "SoftMax"


# ## 3. LogSoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# $\text{logsoftmax}(x)_i = \log\text{softmax}(x)_i = x_i - \log {\sum_j \exp x_j}$
# 
# The main goal of this layer is to be used in computation of log-likelihood loss.

# In[7]:


class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # Вычитаем максимальное значение для численной стабильности
        shifted_input = input - np.max(input, axis=1, keepdims=True)
        exp_input = np.exp(shifted_input)
        self.output = shifted_input - np.log(np.sum(exp_input, axis=1, keepdims=True))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        # Вычисляем softmax
        shifted_input = input - np.max(input, axis=1, keepdims=True)
        exp_input = np.exp(shifted_input)
        softmax = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        
        self.gradInput = gradOutput - softmax * np.sum(gradOutput, axis=1, keepdims=True)
        return self.gradInput
    
    def __repr__(self):
        return "LogSoftMax"


# ## 4. Batch normalization
# One of the most significant recent ideas that impacted NNs a lot is [**Batch normalization**](http://arxiv.org/abs/1502.03167). The idea is simple, yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the way through NN. This improves the convergence for deep models letting it train them for days but not weeks. **You are** to implement the first part of the layer: features normalization. The second part (`ChannelwiseScaling` layer) is implemented below.
# 
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# The layer should work as follows. While training (`self.training == True`) it transforms input as $$y = \frac{x - \mu}  {\sqrt{\sigma + \epsilon}}$$
# where $\mu$ and $\sigma$ - mean and variance of feature values in **batch** and $\epsilon$ is just a small number for numericall stability. Also during training, layer should maintain exponential moving average values for mean and variance: 
# ```
#     self.moving_mean = self.moving_mean * alpha + batch_mean * (1 - alpha)
#     self.moving_variance = self.moving_variance * alpha + batch_variance * (1 - alpha)
# ```
# During testing (`self.training == False`) the layer normalizes input using moving_mean and moving_variance. 
# 
# Note that decomposition of batch normalization on normalization itself and channelwise scaling here is just a common **implementation** choice. In general "batch normalization" always assumes normalization + scaling.

# In[8]:


class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None 
        self.moving_variance = None
        
    def updateOutput(self, input):
        if self.moving_mean is None:
            self.moving_mean = np.zeros(input.shape[1])
            self.moving_variance = np.ones(input.shape[1])

        if self.training:
            mean = np.mean(input, axis=0)
            var = np.var(input, axis=0)
            
            # Обновление бегущих средних
            self.moving_mean = self.moving_mean * self.alpha + mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + var * (1 - self.alpha)
            
            # Используем текущие значения для нормализации
            self.current_mean = mean
            self.current_var = var
        else:
            # В режиме оценки используем бегущие средние
            self.current_mean = self.moving_mean
            self.current_var = self.moving_variance

        self.std = np.sqrt(self.current_var + self.EPS)
        self.normalized = (input - self.current_mean) / self.std
        self.output = self.normalized
        
        return self.output

    def updateGradInput(self, input, gradOutput):
        batch_size = input.shape[0]
        
        # Вычисляем градиенты используя цепное правило
        std_inv = 1. / self.std
        centered_input = input - self.current_mean
        
        # Градиент относительно нормализованного входа
        grad_normalized = gradOutput
        
        # Градиент относительно variance
        grad_var = np.sum(grad_normalized * centered_input * -0.5 * std_inv**3, axis=0)
        
        # Градиент относительно mean
        grad_mean = np.sum(grad_normalized * -std_inv, axis=0) +                    grad_var * np.mean(-2. * centered_input, axis=0)
        
        # Градиент относительно input
        self.gradInput = grad_normalized * std_inv +                         grad_var * 2 * centered_input / batch_size +                         grad_mean / batch_size
        
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"


# In[9]:


class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"


# Practical notes. If BatchNormalization is placed after a linear transformation layer (including dense layer, convolutions, channelwise scaling) that implements function like `y = weight * x + bias`, than bias adding become useless and could be omitted since its effect will be discarded while batch mean subtraction. If BatchNormalization (followed by `ChannelwiseScaling`) is placed before a layer that propagates scale (including ReLU, LeakyReLU) followed by any linear transformation layer than parameter `gamma` in `ChannelwiseScaling` could be freezed since it could be absorbed into the linear transformation layer.

# ## 5. Dropout
# Implement [**dropout**](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). The idea and implementation is really simple: just multimply the input by $Bernoulli(p)$ mask. Here $p$ is probability of an element to be zeroed.
# 
# This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.
# 
# While training (`self.training == True`) it should sample a mask on each iteration (for every batch), zero out elements and multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values which will be in test mode. When testing this module should implement identity transform i.e. `self.output = input`.
# 
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**

# In[10]:


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
        if self.training:
            self.mask = (np.random.random_sample(input.shape) > self.p) / (1 - self.p)
            self.output = input * self.mask
        else:
            self.output = input
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask
        else:
            self.gradInput = gradOutput
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"


# # Activation functions

# Here's the complete example for the **Rectified Linear Unit** non-linearity (aka **ReLU**): 

# In[11]:


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"


# ## 6. Leaky ReLU
# Implement [**Leaky Rectified Linear Unit**](http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs). Expriment with slope. 

# In[12]:


class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.where(input > 0, input, input * self.slope)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, gradOutput, gradOutput * self.slope)
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"


# ## 7. ELU
# Implement [**Exponential Linear Units**](http://arxiv.org/abs/1511.07289) activations.

# In[13]:


class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def updateOutput(self, input):
        self.output = np.where(input > 0, input, 
                                 self.alpha * (np.exp(input) - 1))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, gradOutput,
                                gradOutput * self.alpha * np.exp(input))
        return self.gradInput
    
    def __repr__(self):
        return "ELU"


# ## 8. SoftPlus
# Implement [**SoftPlus**](https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29) activations. Look, how they look a lot like ReLU.

# In[14]:


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.log1p(np.exp(input))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * (1 / (1 + np.exp(-input)))
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"


# # Criterions

# Criterions are used to score the models answers. 

# In[15]:


class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None
        
    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function 
            associated to the criterion and return the result.
            
            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result. 

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)
    
    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput   

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"


# The **MSECriterion**, which is basic L2 norm usually used for regression, is implemented here for you.
# - input:   **`batch_size x n_feats`**
# - target: **`batch_size x n_feats`**
# - output: **scalar**

# In[16]:


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"


# ## 9. Negative LogLikelihood criterion (numerically unstable)
# You task is to implement the **ClassNLLCriterion**. It should implement [multiclass log loss](http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss). Nevertheless there is a sum over `y` (target) in that formula, 
# remember that targets are one-hot encoded. This fact simplifies the computations a lot. Note, that criterions are the only places, where you divide by batch size. Also there is a small hack with adding small number to probabilities to avoid computing log(0).
# - input:   **`batch_size x n_feats`** - probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**
# 
# 

# In[17]:


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def updateOutput(self, input, target): 
        # Убеждаемся, что входные вероятности больше EPS
        input_clipped = np.clip(input, self.EPS, 1 - self.EPS)
        self.output = -np.sum(target * np.log(input_clipped)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        input_clipped = np.clip(input, self.EPS, 1 - self.EPS)
        self.gradInput = -(target / input_clipped) / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"


# ## 10. Negative LogLikelihood criterion (numerically stable)
# - input:   **`batch_size x n_feats`** - log probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**
# 
# Task is similar to the previous one, but now the criterion input is the output of log-softmax layer. This decomposition allows us to avoid problems with computation of forward and backward of log().

# In[18]:


class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def updateOutput(self, input, target):
        self.output = -np.sum(input * target) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput = -target / input.shape[0]
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"


# # Optimizers

# ### SGD optimizer with momentum
# - `variables` - list of lists of variables (one list per layer)
# - `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
# - `config` - dict with optimization parameters (`learning_rate` and `momentum`)
# - `state` - dict with optimizator state (used to save accumulated gradients)

# In[19]:


def sgd_momentum(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})
    
    var_index = 0 
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            
            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)
            
            current_var -= old_grad
            var_index += 1     


# ## 11. [Adam](https://arxiv.org/pdf/1412.6980.pdf) optimizer
# - `variables` - list of lists of variables (one list per layer)
# - `gradients` - list of lists of current gradients (same structure as for `variables`, one array for each var)
# - `config` - dict with optimization parameters (`learning_rate`, `beta1`, `beta2`, `epsilon`)
# - `state` - dict with optimizator state (used to save 1st and 2nd moment for vars)
# 
# Formulas for optimizer:
# 
# Current step learning rate: $$\text{lr}_t = \text{learning_rate} * \frac{\sqrt{1-\beta_2^t}} {1-\beta_1^t}$$
# First moment of var: $$\mu_t = \beta_1 * \mu_{t-1} + (1 - \beta_1)*g$$ 
# Second moment of var: $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2)*g*g$$
# New values of var: $$\text{variable} = \text{variable} - \text{lr}_t * \frac{m_t}{\sqrt{v_t} + \epsilon}$$

# In[20]:


def adam_optimizer(variables, gradients, config, state):  
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()
    
    var_index = 0 
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients): 
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))
            
            np.add(config['beta1'] * var_first_moment,
                   (1 - config['beta1']) * current_grad,
                   out=var_first_moment)

            # Update biased second raw moment estimate
            np.add(config['beta2'] * var_second_moment,
                   (1 - config['beta2']) * np.square(current_grad),
                   out=var_second_moment)

            # Compute bias-corrected first moment estimate
            m_hat = var_first_moment / (1 - config['beta1']**state['t'])

            # Compute bias-corrected second raw moment estimate
            v_hat = var_second_moment / (1 - config['beta2']**state['t'])

            # Update parameters
            current_var -= lr_t * m_hat / (np.sqrt(v_hat) + config['epsilon'])
            
            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1


# # Layers for advanced track homework
# You **don't need** to implement it if you are working on `homework_main-basic.ipynb`

# ## 12. Conv2d [Advanced]
# - input:   **`batch_size x in_channels x h x w`**
# - output: **`batch_size x out_channels x h x w`**
# 
# You should implement something like pytorch `Conv2d` layer with `stride=1` and zero-padding outside of image using `scipy.signal.correlate` function.
# 
# Practical notes:
# - While the layer name is "convolution", the most of neural network frameworks (including tensorflow and pytorch) implement operation that is called [correlation](https://en.wikipedia.org/wiki/Cross-correlation#Cross-correlation_of_deterministic_signals) in signal processing theory. So **don't use** `scipy.signal.convolve` since it implements [convolution](https://en.wikipedia.org/wiki/Convolution#Discrete_convolution) in terms of signal processing.
# - It may be convenient to use `skimage.util.pad` for zero-padding.
# - It's rather ok to implement convolution over 4d array using 2 nested loops: one over batch size dimension and another one over output filters dimension
# - Having troubles with understanding how to implement the layer? 
#  - Check the last year video of lecture 3 (starting from ~1:14:20)
#  - May the google be with you

# In[21]:


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size
       
        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        pad_size = self.kernel_size // 2
        batch_size, channels, height, width = input.shape
        
        # Padding input
        padded_input = np.pad(input, 
                            ((0,0), (0,0), (pad_size,pad_size), (pad_size,pad_size)), 
                            mode='constant')
        
        self.output = np.zeros((batch_size, self.out_channels, height, width))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    self.output[b,oc] += sp.signal.correlate2d(
                        padded_input[b,ic], 
                        self.W[oc,ic], 
                        mode='valid'
                    )
                self.output[b,oc] += self.b[oc]
                
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        batch_size = input.shape[0]
        
        self.gradInput = np.zeros_like(input)
        padded_gradOutput = np.pad(gradOutput,
                                 ((0,0), (0,0), (pad_size,pad_size), (pad_size,pad_size)),
                                 mode='constant')
        
        for b in range(batch_size):
            for ic in range(self.in_channels):
                for oc in range(self.out_channels):
                    self.gradInput[b,ic] += sp.signal.correlate2d(
                        padded_gradOutput[b,oc],
                        np.rot90(self.W[oc,ic], 2),
                        mode='valid'
                    )
        
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        batch_size = input.shape[0]
        
        padded_input = np.pad(input,
                            ((0,0), (0,0), (pad_size,pad_size), (pad_size,pad_size)),
                            mode='constant')
        
        for oc in range(self.out_channels):
            self.gradb[oc] = gradOutput[:,oc].sum()
            for ic in range(self.in_channels):
                for b in range(batch_size):
                    self.gradW[oc,ic] += sp.signal.correlate2d(
                        padded_input[b,ic],
                        gradOutput[b,oc],
                        mode='valid'
                    )
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q


# ## 13. MaxPool2d [Advanced]
# - input:   **`batch_size x n_input_channels x h x w`**
# - output: **`batch_size x n_output_channels x h // kern_size x w // kern_size`**
# 
# You are to implement simplified version of pytorch `MaxPool2d` layer with stride = kernel_size. Please note, that it's not a common case that stride = kernel_size: in AlexNet and ResNet kernel_size for max-pooling was set to 3, while stride was set to 2. We introduce this restriction to make implementation simplier.
# 
# Practical notes:
# - During forward pass what you need to do is just to reshape the input tensor to `[n, c, h / kern_size, kern_size, w / kern_size, kern_size]`, swap two axes and take maximums over the last two dimensions. Reshape + axes swap is sometimes called space-to-batch transform.
# - During backward pass you need to place the gradients in positions of maximal values taken during the forward pass
# - In real frameworks the indices of maximums are stored in memory during the forward pass. It is cheaper than to keep the layer input in memory and recompute the maximums.

# In[22]:


class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None
                    
    def updateOutput(self, input):
        batch_size, channels, height, width = input.shape
        out_height = height // self.kernel_size
        out_width = width // self.kernel_size
        
        # Reshape для удобного max pooling
        reshaped = input.reshape(batch_size, channels, 
                               out_height, self.kernel_size,
                               out_width, self.kernel_size)
        
        # Сохраняем индексы максимумов для backward pass
        self.max_indices = reshaped.max(axis=5).max(axis=3)
        self.output = self.max_indices
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        batch_size, channels, height, width = input.shape
        out_height = height // self.kernel_size
        out_width = width // self.kernel_size
        
        self.gradInput = np.zeros_like(input)
        
        # Восстанавливаем градиенты в позициях максимумов
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.kernel_size
                        w_start = w * self.kernel_size
                        
                        input_slice = input[b,c,
                                          h_start:h_start+self.kernel_size,
                                          w_start:w_start+self.kernel_size]
                        
                        mask = input_slice == input_slice.max()
                        self.gradInput[b,c,
                                     h_start:h_start+self.kernel_size,
                                     w_start:w_start+self.kernel_size] = \
                            mask * gradOutput[b,c,h,w]
                            
        return self.gradInput
    
    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' %(self.kernel_size, self.kernel_size)
        return q


# ### Flatten layer
# Just reshapes inputs and gradients. It's usually used as proxy layer between Conv2d and Linear.

# In[23]:


class Flatten(Module):
    def __init__(self):
         super(Flatten, self).__init__()
    
    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput
    
    def __repr__(self):
        return "Flatten"


# In[ ]:




