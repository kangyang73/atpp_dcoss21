import tensorflow as tf
import numpy as np
import os
import decorated_options as Deco
from .utils import create_dir, MAE, Recall_K

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def softplus(x):
    return np.log1p(np.exp(x))

class AppTimePre:

    @Deco.optioned()
    def __init__(self, sess, app_categories, location_number, user_number, sex_max, age_max, batch_size,
                 learning_rate, momentum, l2_penalty, embed_size, loc_embed_size, user_embed_size, sex_embed_size, age_embed_size,
                 float_type, bptt, seed, scope, save_dir, decay_steps, decay_rate,
                 device_gpu, device_cpu, cpu_only,
                 Wt, Wem, Wel, Weu, Wes, Wea, Wh, bh, wt, Wa, Va, Vt, ba, bt, top_k):
        self.HIDDEN_LAYER_SIZE = Wh.shape[0]
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.MOMENTUM = momentum
        self.L2_PENALTY = l2_penalty
        self.EMBED_SIZE = embed_size
        self.LOC_EMBED_SIZE = loc_embed_size
        self.USER_EMBED_SIZE = user_embed_size
        self.SEX_EMBED_SIZE = sex_embed_size
        self.AGE_EMBED_SIZE = age_embed_size

        self.BPTT = bptt
        self.SAVE_DIR = save_dir
        self.TOP_K = top_k
        self.APP_CATEGORIES = app_categories
        self.LOCATION_NUMBER = location_number
        self.USER_NUMBER = user_number
        self.SEX_MAX = sex_max
        self.AGE_MAX = age_max

        self.FLOAT_TYPE = float_type
        self.DEVICE_CPU = device_cpu
        self.DEVICE_GPU = device_gpu

        self.sess = sess
        self.seed = seed
        self.last_epoch = 0

        self.rs = np.random.RandomState(seed)

        with tf.compat.v1.variable_scope(scope):
            with tf.device(device_gpu if not cpu_only else device_cpu):
                # Input variables
                with tf.compat.v1.variable_scope('inputs'):
                    self.apps_in = tf.compat.v1.placeholder(tf.int32, [None, self.BPTT], name='apps_in')
                    self.times_in = tf.compat.v1.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_in')
                    self.location_in = tf.compat.v1.placeholder(tf.int32, [None, self.BPTT], name='location_in')
                    self.userid_in = tf.compat.v1.placeholder(tf.int32, [None], name='userid_in')
                    self.sex_in = tf.compat.v1.placeholder(tf.int32, [None], name='sex_in')
                    self.age_in = tf.compat.v1.placeholder(tf.int32, [None], name='age_in')
                    self.batch_num_apps = tf.compat.v1.placeholder(self.FLOAT_TYPE, [], name='bptt_apps')

                    self.apps_out = tf.compat.v1.placeholder(tf.int32, [None, self.BPTT], name='apps_out')
                    self.times_out = tf.compat.v1.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_out')

                    self.inf_batch_size = tf.shape(self.apps_in)[0]  # max batch_size

                # Make variables
                with tf.compat.v1.variable_scope('hidden_state'):
                    self.Wt = tf.compat.v1.get_variable(name='Wt', shape=(1, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wt))
                    self.Wem = tf.compat.v1.get_variable(name='Wem', shape=(self.APP_CATEGORIES, self.EMBED_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wem(self.APP_CATEGORIES)))
                    self.Wel = tf.compat.v1.get_variable(name='Wel', shape=(self.LOCATION_NUMBER, self.LOC_EMBED_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wel(self.LOCATION_NUMBER)))
                    self.Weu = tf.compat.v1.get_variable(name='Weu', shape=(self.USER_NUMBER, self.USER_EMBED_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Weu(self.USER_NUMBER)))
                    self.Wes = tf.compat.v1.get_variable(name='Wes', shape=(self.SEX_MAX, self.SEX_EMBED_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wes(self.SEX_MAX)))
                    self.Wea = tf.compat.v1.get_variable(name='Wea', shape=(self.AGE_MAX, self.AGE_EMBED_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wea(self.AGE_MAX)))

                    self.Wh = tf.compat.v1.get_variable(name='Wh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wh))
                    self.bh = tf.compat.v1.get_variable(name='bh', shape=(1, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(bh))

                with tf.compat.v1.variable_scope('output'):
                    self.Vt = tf.compat.v1.get_variable(name='Vt', shape=(self.HIDDEN_LAYER_SIZE, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Vt))
                    self.bt = tf.compat.v1.get_variable(name='bt', shape=(1, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(bt))
                    self.wt = tf.compat.v1.get_variable(name='wt', shape=(1, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(wt))

                    self.Wa = tf.compat.v1.get_variable(name='Wa', shape=(self.EMBED_SIZE, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wa))
                    self.Va = tf.compat.v1.get_variable(name='Va', shape=(self.HIDDEN_LAYER_SIZE, self.APP_CATEGORIES), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Va(self.APP_CATEGORIES)))
                    self.ba = tf.compat.v1.get_variable(name='ba', shape=(1, self.APP_CATEGORIES), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(ba(self.APP_CATEGORIES)))

                self.all_vars = [self.Wt, self.Wem, self.Wel, self.Weu, self.Wes, self.Wea, self.Wh, self.bh,
                                 self.wt, self.Wa, self.Va, self.Vt, self.bt, self.ba]

                self.initial_state = state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE], dtype=self.FLOAT_TYPE, name='initial_state')
                self.initial_time = last_time = tf.zeros((self.inf_batch_size,), dtype=self.FLOAT_TYPE, name='initial_time')
                self.loss = 0.0
                ones_2d = tf.ones((self.inf_batch_size, 1), dtype=self.FLOAT_TYPE)
                self.hidden_states = []
                with tf.name_scope('GRU_layer'):
                    for i in range(self.BPTT):
                        apps_embedded = tf.nn.embedding_lookup(self.Wem, tf.math.mod(self.apps_in[:, i] - 1, self.APP_CATEGORIES))
                        time = self.times_in[:, i]
                        delta_t_prev = tf.expand_dims(time - last_time, axis=-1)
                        last_time = time
                        time_2d = tf.expand_dims(time, axis=-1)
                        type_delta_t = True
                        new_state = tf.tanh(
                            tf.matmul(state, self.Wh) +
                            tf.matmul(apps_embedded, self.Wa) +
                            (tf.matmul(delta_t_prev, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                            tf.matmul(ones_2d, self.bh),
                            name='hidden_state'
                        )
                        state = tf.where(self.apps_in[:, i] > 0, new_state, state)
                        self.hidden_states.append(state)

                self.influence_vectors = []
                self.attention_vectors = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE], dtype=self.FLOAT_TYPE)
                with tf.name_scope('tempotal_app_attention_layer'):
                    app_bptt_next_embedded = tf.nn.embedding_lookup(self.Wem, tf.math.mod(self.apps_out[:, self.BPTT - 1] - 1, self.APP_CATEGORIES))
                    for h_state in self.hidden_states:
                        mul = tf.tanh(tf.multiply(h_state, app_bptt_next_embedded))
                        self.influence_vectors.append(mul)
                    for h_state, influence in zip(self.hidden_states, self.influence_vectors):
                        attention_tmp = tf.multiply(h_state, influence)
                        self.attention_vectors = tf.add(self.attention_vectors, attention_tmp)  # soft attention mechanism

                with tf.name_scope('DNN_layer'):
                    userid_embedded = tf.nn.embedding_lookup(self.Weu, tf.math.mod(self.userid_in - 1, self.USER_NUMBER))
                    sex_embedded = tf.nn.embedding_lookup(self.Wes, tf.math.mod(self.sex_in - 1, self.SEX_MAX))
                    age_embedded = tf.nn.embedding_lookup(self.Wea, tf.math.mod(self.age_in - 1, self.AGE_MAX))

                    temp = tf.concat([userid_embedded, sex_embedded, age_embedded], axis=1)
                    user_vector_tmp = tf.layers.dense(temp, self.EMBED_SIZE * 0.5)
                    user_vector_t = tf.layers.dense(user_vector_tmp, self.EMBED_SIZE)
                    self.user_vector = tf.layers.dense(user_vector_t, self.EMBED_SIZE)

                with tf.name_scope("combination_layer"):
                    location_embedded_tmp = tf.nn.embedding_lookup(self.Wel, tf.math.mod(self.location_in[:, self.BPTT - 1] - 1, self.LOCATION_NUMBER))
                    # location_embedded = tf.layers.dense(location_embedded_tmp, self.EMBED_SIZE)
                    self.temporal_app_vector = tf.multiply(self.attention_vectors, self.user_vector)
                    concat_vector = tf.concat([self.temporal_app_vector, location_embedded_tmp], axis=1)
                    # concat_vector = tf.multiply(self.temporal_app_vector, location_embedded_tmp)

                    fully_connected_vector1 = tf.layers.dense(concat_vector, self.EMBED_SIZE * 3)
                    fully_connected_vector2 = tf.layers.dense(fully_connected_vector1, self.EMBED_SIZE * 2)
                    self.final_state = tf.layers.dense(fully_connected_vector2, self.EMBED_SIZE)

                self.app_preds = []
                self.time_preds = []
                with tf.name_scope('loss_layer'):
                    i = self.BPTT - 1
                    time = self.times_in[:, i]
                    time_next = self.times_out[:, i]

                    delta_t_next = tf.expand_dims(time_next - time, axis=-1)
                    time_expand = tf.expand_dims(time, axis=-1)
                    time_next_expand = tf.expand_dims(time_next, axis=-1)
                    base_intensity = tf.matmul(ones_2d, self.bt)
                    wt_soft_plus = tf.nn.softplus(self.wt)  # softplus: log(exp(x) + 1)

                    log_lambda_ = (tf.matmul(self.final_state, self.Vt) + (delta_t_next * wt_soft_plus) + base_intensity)
                    # log_lambda_ = (tf.matmul(self.final_state, self.Vt) + base_intensity)
                    # lambda_ = tf.exp(tf.minimum(1.0, log_lambda_), name='lambda_')

                    # self.apps_pred = tf.nn.softmax(tf.minimum(50.0, tf.matmul(self.final_state, self.Va) + ones_2d * self.ba), name='Predcition_apps')
                    self.apps_pred = tf.nn.softmax(tf.matmul(self.final_state, self.Va) + ones_2d * self.ba, name='Predcition_apps')
                    self.times_pred = time_expand + log_lambda_

                    # mark_LL = tf.expand_dims(tf.math.log(tf.maximum(1e-6, tf.gather_nd(self.apps_pred, tf.concat([tf.expand_dims(tf.range(self.inf_batch_size), -1), tf.expand_dims(tf.math.mod(self.apps_out[:, i] - 1, self.APP_CATEGORIES), -1)], axis=1, name='Pr_next_app')))), axis=-1, name='log_Pr_next_app')
                    mark_LL = tf.expand_dims(tf.math.log(tf.gather_nd(self.apps_pred, tf.concat([tf.expand_dims(tf.range(self.inf_batch_size), -1), tf.expand_dims(tf.math.mod(self.apps_out[:, 0] - 1, self.APP_CATEGORIES), -1)], axis=1, name='Pr_next_app'))), axis=-1, name='log_Pr_next_app')
                    time_LL = tf.subtract(self.times_pred, time_next_expand)
                    step_LL = (time_LL + mark_LL)

                    self.loss -= tf.reduce_sum(tf.where(self.apps_in[:, i] > 0, tf.squeeze(step_LL) / self.batch_num_apps, tf.ones(shape=(self.inf_batch_size,)) * 0.01))
                    # self.loss -= tf.reduce_mean(tf.where(self.apps_in[:, i] > 0, tf.squeeze(step_LL) / self.batch_num_apps, tf.ones(shape=(self.inf_batch_size,)) * 0.01)) \
                    #              + self.L2_PENALTY * (tf.nn.l2_loss(self.Wt) + tf.nn.l2_loss(self.Wem) + tf.nn.l2_loss(self.Wel) + tf.nn.l2_loss(self.Weu) + tf.nn.l2_loss(self.Wes) \
                    #              + tf.nn.l2_loss(self.Wea) + tf.nn.l2_loss(self.Wh) + tf.nn.l2_loss(self.bh) + tf.nn.l2_loss(self.wt) + tf.nn.l2_loss(self.Wa)\
                    #              + tf.nn.l2_loss(self.Va) + tf.nn.l2_loss(self.Vt) + tf.nn.l2_loss(self.bt) + tf.nn.l2_loss(self.ba))


                with tf.device(device_cpu):
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.learning_rate = tf.compat.v1.train.inverse_time_decay(self.LEARNING_RATE, global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)

                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.MOMENTUM)
                self.gvs = self.optimizer.compute_gradients(self.loss)
                grads, vars_ = list(zip(*self.gvs))
                self.norm_grads, self.global_norm = tf.clip_by_global_norm(grads, 10.0)
                capped_gvs = list(zip(self.norm_grads, vars_))
                self.update = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
                self.tf_init = tf.compat.v1.global_variables_initializer()

    def initialize(self, finalize=False):
        """Initialize the global trainable variables."""
        self.sess.run(self.tf_init)
        if finalize:
            self.sess.graph.finalize()

    def train(self, training_data, num_epochs=1, restart=False):

        create_dir(self.SAVE_DIR)
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        if ckpt and restart:
            print('Restoring from {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        train_app_in_seq = training_data['train_app_in_seq']
        train_location_in_seq = training_data['train_location_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']

        train_app_out_seq = training_data['train_app_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']

        train_user_info = training_data['train_user_info']

        idxes = list(range(len(train_app_in_seq)))
        n_batches = len(idxes) // self.BATCH_SIZE

        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            self.rs.shuffle(idxes)
            print("Starting epoch...", epoch)
            total_loss = 0.0

            for batch_idx in range(n_batches):

                batch_idxes = idxes[batch_idx * self.BATCH_SIZE: (batch_idx + 1) * self.BATCH_SIZE]
                batch_app_train_in = train_app_in_seq[batch_idxes, :]
                batch_app_train_out = train_app_out_seq[batch_idxes, :]
                batch_location_train_in = train_location_in_seq[batch_idxes, :]
                # batch_location_train_out = train_location_out_seq[batch_idxes, :]
                batch_time_train_in = train_time_in_seq[batch_idxes, :]
                batch_time_train_out = train_time_out_seq[batch_idxes, :]

                batch_train_user_info = train_user_info[batch_idxes, :]  # BATCH_SIZE * 3

                cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                batch_loss = 0.0
                batch_num_apps = np.sum(batch_app_train_in > 0)
                for bptt_idx in range(0, len(batch_app_train_in[0]) - self.BPTT, self.BPTT):
                    bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
                    bptt_app_in = batch_app_train_in[:, bptt_range]
                    bptt_app_out = batch_app_train_out[:, bptt_range]
                    bptt_time_in = batch_time_train_in[:, bptt_range]
                    bptt_time_out = batch_time_train_out[:, bptt_range]

                    bptt_location_in = batch_location_train_in[:, bptt_range]
                    bptt_userid_in = batch_train_user_info[:, 0]
                    bptt_sex_in = batch_train_user_info[:, 1]
                    bptt_age_in = batch_train_user_info[:, 2]

                    if np.all(bptt_app_in[:, 0] == 0):
                        break

                    if bptt_idx > 0:
                        initial_time = batch_time_train_in[:, bptt_idx - 1]
                    else:
                        initial_time = np.zeros(batch_time_train_in.shape[0])

                    feed_dict = {
                        self.initial_state: cur_state,
                        self.initial_time: initial_time,
                        self.apps_in: bptt_app_in,
                        self.apps_out: bptt_app_out,
                        self.times_in: bptt_time_in,
                        self.times_out: bptt_time_out,
                        self.batch_num_apps: batch_num_apps,
                        self.location_in: bptt_location_in,
                        self.userid_in: bptt_userid_in,
                        self.sex_in: bptt_sex_in,
                        self.age_in: bptt_age_in,
                    }

                    _, cur_state, loss_, step = self.sess.run([self.update, self.final_state, self.loss, self.global_step], feed_dict=feed_dict)
                    # print("loss", loss_)
                    # print("gvsssssssssssss", gvs)
                    batch_loss += loss_

                total_loss += batch_loss
                # if batch_idx % 10 == 0:
                print('Loss during batch {} batch_loss = {:.5f}, lr = {:.9f}'.format(batch_idx, batch_loss, self.sess.run(self.learning_rate)))

            print('One epoch avarage loss = {:.5f}, new learn rate = {:.9f}, global_step = {}'
                  .format(total_loss / n_batches, self.sess.run(self.learning_rate), self.sess.run(self.global_step)))

        checkpoint_path = os.path.join(self.SAVE_DIR, 'model.ckpt')
        saver.save(self.sess, checkpoint_path, global_step=self.global_step)
        print('Model saved at {}'.format(checkpoint_path))

        self.last_epoch += num_epochs

    def restore(self):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        print('Loading the model from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, app_in_seq, time_in_seq, app_out_seq, time_out_seq, location_in_seq, user_info):
        all_app_preds = []
        all_time_preds = []
        cur_state = np.zeros((len(app_in_seq), self.HIDDEN_LAYER_SIZE))

        for bptt_idx in range(0, len(app_in_seq[0]) - self.BPTT, self.BPTT):
            bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
            bptt_app_in = app_in_seq[:, bptt_range]
            bptt_time_in = time_in_seq[:, bptt_range]

            bptt_location_in = location_in_seq[:, bptt_range]
            bptt_userid_in = user_info[:, 0]
            bptt_sex_in = user_info[:, 1]
            bptt_age_in = user_info[:, 2]

            bptt_app_out = app_out_seq[:, bptt_range]
            bptt_time_out = time_out_seq[:, bptt_range]

            if bptt_idx > 0:
                initial_time = app_in_seq[:, bptt_idx - 1]
            else:
                initial_time = np.zeros(bptt_time_in.shape[0])

            feed_dict = {
                self.initial_state: cur_state,
                self.initial_time: initial_time,
                self.apps_in: bptt_app_in,
                self.times_in: bptt_time_in,
                self.location_in: bptt_location_in,
                self.userid_in: bptt_userid_in,
                self.sex_in: bptt_sex_in,
                self.age_in: bptt_age_in,

                self.apps_out: bptt_app_out,
                self.times_out: bptt_time_out
            }

            bptt_apps_pred, bptt_times_pred, cur_state = self.sess.run([self.apps_pred, self.times_pred, self.final_state], feed_dict=feed_dict)

            all_app_preds.append(bptt_apps_pred)
            all_time_preds.append(bptt_times_pred)

        app_t = np.asarray(all_app_preds).swapaxes(0, 1)
        time_t = np.asarray(all_time_preds).reshape(app_t.shape[0], app_t.shape[1])
        return time_t, app_t

    def eval(self, time_preds, time_true, app_preds, app_true):

        mae, total_valid = MAE(time_preds, time_true, app_true)
        recall = Recall_K(app_preds, app_true, self.TOP_K)

        return mae, recall, total_valid

    def predict_test(self, data):
        return self.predict(app_in_seq=data['test_app_in_seq'],
                            time_in_seq=data['test_time_in_seq'],
                            location_in_seq=data['test_location_in_seq'],
                            user_info=data['test_user_info'],
                            app_out_seq=data['test_app_out_seq'],
                            time_out_seq=data['test_time_out_seq'])

    def predict_train(self, data):
        return self.predict(app_in_seq=data['train_app_in_seq'],
                            time_in_seq=data['train_time_in_seq'],
                            location_in_seq=data['train_location_in_seq'],
                            user_info=data['train_user_info'],
                            app_out_seq=data['train_app_out_seq'],
                            time_out_seq=data['train_time_out_seq'])
