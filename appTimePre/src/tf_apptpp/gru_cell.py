import tensorflow as tf

class TPPRExpMarkedCellStacked(tf.contrib.rnn.RNNCell):

    def __init__(self, hidden_state_size, output_size, tf_dtype,
                 Wem, Wh, Wa, Wt, bh):
        self._hidden_state_size = hidden_state_size
        self._output_size = output_size
        self.tf_dtype = tf_dtype

        self.Wem = Wem
        self.Wh = Wh
        self.Wa = Wa
        self.Wt = Wt
        self.bh = bh

    def __call__(self, inp, h_prev):
        apps_in, t_delta = inp  # input
        APP_CATEGORIES = tf.shape(self.Wem)[0]

        apps_embedded = tf.nn.embedding_lookup(self.Wem, tf.math.mod(apps_in - 1, APP_CATEGORIES))
        inf_batch_size = tf.shape(apps_in)[0]
        ones_2d = tf.ones((inf_batch_size, 1), dtype=self.tf_dtype)

        h_next = tf.nn.tanh(
            tf.einsum('aij,aj->ai', self.Wh, h_prev) +
            tf.einsum('aij,aj->ai', self.Wa, apps_embedded) +
            tf.einsum('aij,aj->ai', self.Wt, t_delta) +
            tf.einsum('aij,aj->ai', self.bh, ones_2d),
            name='h_next'
        )

        return ((h_next), h_next)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._hidden_state_size
