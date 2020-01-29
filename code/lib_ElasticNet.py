import tensorflow as tf

class ElasticNet:
    def __init__(self, num_predictors, alpha, lambda_, A = None, b = None, seed = None):
        init = tf.random_normal_initializer(seed = seed)
        if A is None:
            A = tf.Variable(init(shape = [num_predictors, 1]), name = 'A')
        if b is None:
            b = tf.Variable(init(shape = [1, 1]), name = 'b')
        self.A = A
        self.b = b
        self.trainable_variables = [self.A, self.b]
        self.proximal_variables = [self.A]
        self.not_prox_variables = [self.b]
        self.alpha = alpha
        self.lambda_ = lambda_
        l1, l2 = self.lambda_alpha_to_l1_l2(alpha, lambda_)
        self.update_l1_l2(l1, l2)
    
    def update_l1_l2(self, l1, l2):
        self.l1 = tf.constant(tf.cast(l1, tf.float32))
        self.l2 = tf.constant(tf.cast(l2, tf.float32))
    
    def lambda_alpha_to_l1_l2(self, alpha, lambda_):
        l1 = alpha * lambda_
        l2 = (1 - alpha) * lambda_
        return l1, l2
    
    def forward(self, x):
        return tf.tensordot(x, self.A, axes = [1, 0]) + self.b

    def regu_l1(self):
        regu_l1 = tf.reduce_sum(tf.abs(self.A))
        return tf.multiply(self.l1, regu_l1)

    def regu_l2(self):
        regu_l2 = tf.reduce_sum(tf.square(self.A))
        return tf.multiply(self.l2, regu_l2)

    def regularization(self):
        return tf.add(self.regu_l1(), self.regu_l2()) 

    def loss(self, x, y):
        return tf.reduce_mean(tf.square(y - self.forward(x)))

    def objective(self, x, y):
        tmp = self.loss(x, y)
        return tf.add(tmp, self.regularization()), tmp

    def proximal_obj(self, x, y):
        tmp = self.loss(x, y)
        return tf.add(tmp, self.regu_l2()), tmp

    def update_lambda(self, lambda_):
        l1, l2 = self.lambda_alpha_to_l1_l2(self.alpha, lambda_)
        self.lambda_ = lambda_
        self.update_l1_l2(l1, l2)
    def copy(self):
        return self.__init__(None, self.alpha, self.lambda_, A = self.A, b = self.b)

class ProximalUpdater:  # it is not called optimizer to distinguish from the Optimizer class in TF2
    def __init__(self, learning_rate = None, line_search = False):
        self.learning_rate = learning_rate
        self.line_search = line_search
        self._t_init = 1
        self._beta = 0.1    
    def prox_l1(self, x, lambda_t):
        sign = tf.math.sign(x)
        x_n_lambda_t = tf.add(x, tf.multiply(sign, tf.negative(lambda_t)))
        # apply soft thresholding
        sign_new = tf.math.sign(x_n_lambda_t)
        sign_preserve = tf.math.equal(sign, sign_new) 
        return tf.add(
            tf.multiply(tf.cast(sign_preserve, tf.float32), x_n_lambda_t), 
            tf.multiply(tf.cast(tf.logical_not(sign_preserve), tf.float32), tf.zeros(x.shape))
        )
    def _prox_update(self, pairs_prox, pairs_not_prox, l1, learning_rate):
        for grad, var in pairs_not_prox:
            var.assign(var - tf.multiply(learning_rate, grad))
        for grad, var in pairs_prox:
            tmp_prox = var - tf.multiply(learning_rate, grad)
            var.assign(self.prox_l1(tmp_prox, tf.multiply(l1, self.learning_rate)))
    def proximal_update(self, pairs_prox, pairs_not_prox, model, x, y):              
        if self.learning_rate is not None and self.line_search is False:
            self._prox_update(pairs_prox, pairs_not_prox, model.l1, self.learning_rate)
        else:
            # TODO: add backtrack line search to determine step size 
            if self.learning_rate is not None:
                t_init = self.learning_rate 
            else:
                t_curr = self._t_init
                lhs, rhs = self.__calc_line_search(t_curr, pairs_prox, pairs_not_prox, model, x, y)
                while lhs > rhs:
                    t_curr = t_curr * self.beta
                    lhs, rhs = self.__calc_line_search(t_curr, pairs_prox, pairs_not_prox, model, x, y)
                self._prox_update(pairs_prox, pairs_not_prox, model.l1, t_curr)
    def __calc_line_search(self, t_curr, pairs_prox, pairs_not_prox, model, x, y):
        model_copy = model.copy()
        self._prox_update(
            zip(grad_prox, model_copy.proximal_variables), 
            zip(grad_not_prox, model_copy.not_prox_variables), 
            model_copy.l1
        )
        t_Gt = model_copy.trainable_variables
        for i in range(len(t_Gt)):
            t_Gt[i] = (model.trainable_variables[i] - model_copy.trainable_variables)
        inner_product = self.__inner_product(
            list(list(zip(*pairs_not_prox))[0]), 
            model_copy.not_prox_variables
        ) + self.__inner_product(
            list(list(zip(*pairs_not_prox))[0]), 
            model_copy.proximal_variables
        )
        norm2_t_Gt = self.__inner_product(t_Gt, t_Gt)
        lhs = model_copy.proximal_obj(x, y)
        rhs = model.proximal_obj(x, y) - inner_product + norm2_t_Gt / t_curr / 2 
        print(f'lhs = {lhs}, rhs = {rhs}')
        del model_copy
        return lhs, rhs
    def __inner_product(self, e1, e2):
        return_val = 0
        for x, y in zip(e1, e2):
            return_val += tf.reduce_sum(tf.multiply(x, y))
        return return_val
    def proximal_train_step(self, model, x, y):
    #     for x, y in dataset:
        with tf.GradientTape() as tape:
            obj, loss = model.proximal_obj(x, y)
        grad = tape.gradient(obj, [model.proximal_variables, model.not_prox_variables])
        grad_prox = grad[0]
        grad_not_prox = grad[1]
        self.proximal_update(
            zip(grad_prox, model.proximal_variables), 
            zip(grad_not_prox, model.not_prox_variables),
            model,
            x, y
        )
        return loss, obj
