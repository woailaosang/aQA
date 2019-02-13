class Opt():
    def __init__(self):
        # mode options
        self.isDebug = False
        self.decoder_mode = 0  # {0:'pointer-generator', 1:'only-generator', 2:'only-copy'}

        # GPU
        self.usegpu = True
        self.gpuid = 1

        # preprocess.py
        self.seed = 3435
        self.totalVocabSize = 50000  # 使用的时候最好用dicts.size()
        self.genVocabSize = 5000
        self.lower = True
        self.shuffle = True
        self.word_embedding_path = "./data/glove.840B.300d.txt"
        self.PAD, self.UNK, self.BOS, self.EOS = 0, 1, 2, 3
        self.PAD_WORD, self.UNK_WORD, self.BOS_WORD, self.EOS_WORD = '<pad>', '<unk>', '<s>', '</s>'

        # Model options
        self.layers = 2
        self.rnn_size = 500  # 包含双向的
        self.word_vec_size = 300
        self.brnn = True

        # Optimization options
        self.batch_size = 6
        self.epochs = 50
        self.start_epoch = 1
        self.param_init = 0.1

        self.dropout = 0.3
        self.max_generator_batches = 32
        self.max_decoder_length = 60
        self.beam_size = 4
        self.n_best = 1

        # learning rate
        self.optim = 'sgd'
        self.learning_rate = 0.1
        self.max_grad_norm = 5
        self.learning_rate_decay = 0.5
        self.start_decay_at = None  # 8

