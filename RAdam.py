import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer

class RAdamOptimizer(optimizer.Optimizer):

    """
    RAdam optimizer : On The Variance Of The Adaptive Learning Rate And Beyond
    https://arxiv.org/abs/1908.03265
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.,
                 use_locking=False,
                 name="RAdam"):

        super(RAdamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = weight_decay

        self._lr_t = None
        self._step_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._weight_decay_t = None

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("step", graph=graph),
                    self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=1.0, name="step", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)
        weight_decay = self._call_if_callable(self._weight_decay)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
        self._weight_decay_t = ops.convert_to_tensor(weight_decay, name="weight_decay")

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        step, beta1_power, beta2_power = self._get_beta_accumulators()
        step = math_ops.cast(step, var.dtype.base_dtype)
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        sma_inf = 2.0 / (1.0 - beta2_t) - 1.0
        sma_t = sma_inf - 2.0 * step * beta2_power / (1.0 - beta2_power)

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + (1.0 - beta1_t) * grad, use_locking=self._use_locking)
        mhat_t = m_t / (1.0 - beta1_power)

        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, beta2_t * v + (1.0 - beta2_t) * math_ops.square(grad), use_locking=self._use_locking)
        vhat_t = math_ops.sqrt(v_t / ((1.0 - beta2_power) + epsilon_t))

        r_t = math_ops.sqrt( ((sma_t - 4.0) * (sma_t - 2.0) * sma_inf) / ((sma_inf - 4.0) * (sma_inf - 2.0) * sma_t) )

        var_t = tf.cond(sma_t >= 5.0, lambda : r_t * mhat_t / (vhat_t + epsilon_t), lambda : mhat_t)

        if self._weight_decay > 0.0:
            var_t += math_ops.cast(self._weight_decay_t, var.dtype.base_dtype) * var

        var_update = state_ops.assign_sub(var, lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]

        return control_flow_ops.group(*updates)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        step, beta1_power, beta2_power = self._get_beta_accumulators()
        step = math_ops.cast(step, var.dtype.base_dtype)
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        sma_inf = 2.0 / (1.0 - beta2_t) - 1.0
        sma_t = sma_inf - 2.0 * step * beta2_power / (1.0 - beta2_power)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)

        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        mhat_t = m_t / (1.0 - beta1_power)

        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)

        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        vhat_t = math_ops.sqrt(v_t / (1.0 - beta2_power) + epsilon_t)

        r_t = math_ops.sqrt( ((sma_t - 4.0) * (sma_t - 2.0) * sma_inf) / ((sma_inf - 4.0) * (sma_inf - 2.0) * sma_t) )

        var_t = tf.cond(sma_t >= 5.0, lambda : r_t * mhat_t / (vhat_t + epsilon_t), lambda : mhat_t)

        if self._weight_decay > 0.0:
            var_t += math_ops.cast(self._weight_decay_t, var.dtype.base_dtype) * var

        var_update = state_ops.assign_sub(var, lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]

        return control_flow_ops.group(*updates)

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies([resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            step, beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_step = step.assign(step + 1.0, use_locking=self._use_locking)
                update_beta1 = beta1_power.assign(beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_step, update_beta1, update_beta2], name=name_scope)
