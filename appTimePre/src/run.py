import click
import tf_apptpp
import tensorflow as tf
import decorated_options as Deco
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

embed_size = 128
loc_embed_size = 14  
user_embed_size = 10  
sex_embed_size = 2
age_embed_size = 7  

hidden_size = 128
batch_size = 16
bptt = 8
top_k = 5
epochs = 10000
seed = 1994
learning_rate = 1e-7

sex_max = 2
age_max = 70

def_opts = Deco.Options(
    save_dir='./save/',
    epochs=epochs,
    restart=True,
    train_eval=False,
    test_eval=True,
    batch_size=batch_size,
    bptt=bptt,
    learning_rate=learning_rate,
    top_k=top_k,
    cpu_only=True,

    float_type=tf.float32,
    seed=seed,

    scope='appTimePre',
    device_gpu='/gpu:0',
    device_cpu='/cpu:0',

    momentum=0.9,
    decay_steps=1000,
    decay_rate=0.001,
    l2_penalty=0.0001,

    embed_size=embed_size,
    loc_embed_size=loc_embed_size,
    user_embed_size=user_embed_size,
    sex_embed_size=sex_embed_size,
    age_embed_size=age_embed_size,

    Wem=lambda app_categories: np.random.RandomState(seed).randn(app_categories, embed_size) * 1e-4,
    Wel=lambda location_number: np.random.RandomState(seed).randn(location_number, loc_embed_size) * 1e-4,
    Weu=lambda user_number: np.random.RandomState(seed).randn(user_number, user_embed_size) * 1e-4,
    Wes=lambda sex_max: np.random.RandomState(seed).randn(sex_max, sex_embed_size) * 1e-4,
    Wea=lambda age_max: np.random.RandomState(seed).randn(age_max, age_embed_size) * 1e-4,

    Wt=np.ones((1, hidden_size)) * 1e-4,
    Vt=np.ones((hidden_size, 1)) * 1e-4,
    bt=np.ones((1, 1)) * 1e-4,
    wt=np.ones((1, 1)) * 1e-4,

    Wh=np.eye(hidden_size),
    bh=np.ones((1, hidden_size)),

    Wa=np.ones((embed_size, hidden_size)) * 1e-4,
    Va=lambda app_categories: np.ones((hidden_size, app_categories)) * 1e-4,
    ba=lambda app_categories: np.ones((1, app_categories)) * 1e-4
)

@click.command()
@click.argument('app_train_file')
@click.argument('location_train_file')
@click.argument('time_train_file')
@click.argument('app_test_file')
@click.argument('location_test_file')
@click.argument('time_test_file')
@click.argument('user_info_file')
@click.option('--save', 'save_dir', help='Which folder to save  to.', default=def_opts.save_dir)
@click.option('--epochs', 'num_epochs', help='How many epochs to train for.', default=def_opts.epochs)
@click.option('--restart', 'restart', help='Can restart from a saved model from the summary folder, if available.', default=def_opts.restart)
@click.option('--train-eval', 'train_eval', help='Should evaluate the model on training data?', default=def_opts.train_eval)
@click.option('--test-eval', 'test_eval', help='Should evaluate the model on test data?', default=def_opts.test_eval)
@click.option('--batch-size', 'batch_size', help='Batch size.', default=def_opts.batch_size)
@click.option('--bptt', 'bptt', help='Series dependence depth.', default=def_opts.bptt)
@click.option('--learning-rate', 'learning_rate', help='Initial learning rate.', default=def_opts.learning_rate)
@click.option('--cpu-only', 'cpu_only', help='Use only the CPU.', default=def_opts.cpu_only)
def cmd(app_train_file, location_train_file, time_train_file, app_test_file, location_test_file, time_test_file, user_info_file,
        save_dir, num_epochs, restart, train_eval, test_eval,
        batch_size, bptt, learning_rate, cpu_only):

    data = tf_apptpp.utils.read_data(
        app_train_file=app_train_file,
        app_test_file=app_test_file,
        location_train_file=location_train_file,
        location_test_file=location_test_file,
        time_train_file=time_train_file,
        time_test_file=time_test_file,
        user_info_file=user_info_file
    )

    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    # describle data
    tf_apptpp.utils.data_stats(data)

    # build model
    apptpp_model = tf_apptpp.apptpp.AppTimePre(
        sess=sess,
        app_categories=data['app_categories'],
        location_number=data['location_number'],
        user_number=data['user_number'],
        sex_max=sex_max,
        age_max=age_max,

        batch_size=batch_size,
        bptt=bptt,
        learning_rate=learning_rate,
        cpu_only=cpu_only,
        _opts=def_opts,
        save_dir=save_dir
    )

    apptpp_model.initialize(finalize=False)
    apptpp_model.train(training_data=data, restart=restart, num_epochs=num_epochs)

    verification_epoch = 1
    if train_eval:
        print('\nEvaluation on training data:')
        mae_arr = []
        recall_arr = []
        for idx, batch_size_data in enumerate(tf_apptpp.utils.train_data_split_batch(data, batch_size, verification_epoch, bptt, seed)):

            train_time_preds, train_app_preds = apptpp_model.predict_train(data=batch_size_data)
            batch_mae, batch_recall, total_valid = apptpp_model.eval(train_time_preds, batch_size_data['train_time_out_seq_pre'], train_app_preds, batch_size_data['train_app_out_seq_pre'])
            mae_arr.append(batch_mae)
            recall_arr.append(batch_recall)

            # if idx % 10 == 0:
            print('** MAE = {:.6f}; valid = {}, Recall_K = {:.6f}'.format(batch_mae, total_valid, batch_recall))

        print('Performances on training data for mae:', np.array(mae_arr).mean())
        print('Performances on training data for recall:', np.array(recall_arr).mean())

    if test_eval:
        print('\nEvaluation on testing data:')
        mae_arr = []
        recall_arr = []
        for idx, batch_size_data in enumerate(tf_apptpp.utils.test_data_split_batch(data, batch_size, verification_epoch, bptt, seed)):

            test_time_preds, test_app_preds = apptpp_model.predict_test(data=batch_size_data)
            batch_mae, batch_recall, total_valid = apptpp_model.eval(test_time_preds, batch_size_data['test_time_out_seq_pre'], test_app_preds, batch_size_data['test_app_out_seq_pre'])
            mae_arr.append(batch_mae)
            recall_arr.append(batch_recall)

            # if idx % 10 == 0:
            print('** idx = {}; MAE = {:.6f}; valid = {}, Recall_K = {:.6f}'.format(idx, batch_mae, total_valid, batch_recall))

        print('Performances on testing data for mae:', np.array(mae_arr).mean())
        print('Performances on testing data for recall:', np.array(recall_arr).mean())

if __name__ == '__main__':

    cmd()
