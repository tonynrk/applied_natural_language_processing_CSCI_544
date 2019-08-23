import sys
import numpy
import tensorflow as tf
import codecs
import re
import pickle

USC_EMAIL = 'khieopra@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = '87d2c7ce44479087'  # TODO(student): You will be given a password via email.
TRAIN_TIME_MINUTES = 9

class DatasetReader(object):

    # TODO(student): You must implement this.
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        """Reads file into dataset, while populating term_index and tag_index.

        Args:
            filename: Path of text file containing sentences and tags. Each line is a
                sentence and each term is followed by "/tag". Note: some terms might
                have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
                separates the tag.
            term_index: dictionary to be populated with every unique term (i.e. before
                the last "/") to point to an integer. All integers must be utilized from
                0 to number of unique terms - 1, without any gaps nor repetitions.
            tag_index: same as term_index, but for tags.

        the _index dictionaries are guaranteed to have no gaps when the method is
        called i.e. all integers in [0, len(*_index)-1] will be used as values.
        You must preserve the no-gaps property!




        Return:
            The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
            each parsedLine is a list: [(termId1, tagId1), (termId2, tagId2), ...]
        """

        def tokenizeforslash(tmp):
            subtmp = 0
            for i in range(len(tmp))[::-1]:
                if tmp[i] == "/":
                    subtmp = i
                    break
            return [tmp[0:subtmp], tmp[subtmp + 1:]]

        term_count = 0
        tag_count = 0

        if len(term_index) > 1:
            term_count = max(term_index.values()) + 1

        if len(tag_index) > 1:
            tag_count = max(tag_index.values()) + 1

        with codecs.open(filename, encoding='utf-8') as file:

            parsedFile = []
            # read all file
            for line in file:
                # give every word to be lowercase

                tokens = re.split('[\s]+', line.strip())
                # tokens = line.lower().strip().split(" ")
                parsedLine = []
                for i in range(len(tokens)):
                    token = tokenizeforslash(tokens[i])
                    term = token[0]
                    tag = token[1]
                    if term not in term_index:
                        term_index[term] = term_count
                        term_count += 1
                    if tag not in tag_index:
                        tag_index[tag] = tag_count
                        tag_count += 1

                    parsedLine.append((term_index[term], tag_index[tag]))

                parsedFile.append(parsedLine)

        return parsedFile

    # TODO(student): You must implement this.
    @staticmethod
    def BuildMatrices(dataset):
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

        Args:
            dataset: Returned by method ReadFile. It is a list (length N) of lists:
                [sentence1, sentence2, ...], where every sentence is a list:
                [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

        Returns:
            Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
                terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
                    indices in dataset[i].
                tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
                    indices in dataset[i].
                lengths: shape (N) int64 numpy array. Entry i contains the length of
                    sentence in dataset[i].

            T is the maximum length. For example, calling as:
                BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
            i.e. with two sentences, first with length 2 and second with length 4,
            should return the tuple:
            (
                [[1, 4, 0, 0],    # Note: 0 padding.
                 [13, 3, 7, 3]],

                [[2, 10, 0, 0],   # Note: 0 padding.
                 [20, 6, 8, 20]], 

                [2, 4]
            )
        """

        length = []
        max_length = 0

        for var in dataset:
            #adding length
            length.append(len(var))
            #find the max length
            if max_length < len(var):
                max_length = len(var)

        length = numpy.array(length,dtype=numpy.int64)

        terms_matrix = numpy.zeros(shape=(len(dataset), max_length),dtype=numpy.int64)

        tags_matrix = numpy.zeros(shape=(len(dataset), max_length),dtype=numpy.int64)


        # term
        for i,data in enumerate(dataset):
            for j,var in enumerate(data):
                terms_matrix[i][j] = var[0]

        # tag
        for i,data in enumerate(dataset):
            for j,var in enumerate(data):
                tags_matrix[i][j] = var[1]

        all_value = [terms_matrix,tags_matrix,length]

        return all_value


    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.

        NOTE: Please do not change this method. The grader will use an identitical
        copy of this method (if you change this, your offline testing will no longer
        match the grader).

        Args:
            train_filename: .txt path containing training data, one line per sentence.
                The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.

        Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries,
            respectively, from term to integer ID and from tag to integer ID. The int
            IDs are used in the numpy matrices.
            The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
                - train_terms: numpy int matrix.
                - train_tags: numpy int matrix.
                - train_lengths: numpy int vector.
                These 3 are identical to what is returned by BuildMatrices().
            The 4th element is a tuple of 3 elements as above, but the data is
            extracted from test_filename.
        """
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        tag_index = {}

        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)

        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype=numpy.int64)
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype=numpy.int64)
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                            (train_terms, train_tags, train_lengths),
                            (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)







class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """Constructor. You can add code but do not remove any code.

        The arguments are arbitrary: when you are training on your own, PLEASE set
        them to the correct values (e.g. from main()).

        Args:
            max_lengths: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).

        You will be passed these arguments by the grader script.
        """
        self.sess = tf.Session()
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')

        #self.state_size = 100

        self.targets = tf.placeholder(tf.int64,[None, self.max_length],'targets')
        self.learning_rate = tf.placeholder_with_default(numpy.array(numpy.array(0.01, dtype='float32')),shape=[] ,name='learn_rate')
        self.is_training = tf.placeholder(tf.bool,[])


    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.
        
        Specifically, the return matrix B will have the following:
            B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        # return tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32)
        return tf.sequence_mask(length_vector, self.max_length, dtype=tf.float32)

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""
        var_dict = {v.name: v for v in tf.global_variables()}
        pickle.dump(self.sess.run(var_dict), open(filename, 'wb'))
        return

    # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        var_values = pickle.load(open(filename))
        assign_ops = [v.assign(var_values[v.name]) for v in tf.global_variables()]
        self.sess.run(assign_ops)
        return

    # TODO(student): You must implement this.
    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).
        
        Please do not change or override self.x nor self.lengths in this function.

        Hint:
            - Use lengths_vector_to_binary_matrix
            - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """
        # TODO(student): make logits an RNN on x.

        # embed_size = 100
        #
        # embeds = tf.get_variable('embeddings', [self.num_terms,embed_size]) # batch_size = 100
        # x_embed = tf.nn.embedding_lookup(embeds, self.x)
        #
        #
        # state_size = 100
        #
        # rnn_cell = tf.keras.layers.SimpleRNNCell(state_size)
        #
        # states = []
        #
        #
        #
        # for i in range(self.max_length):
        #     cur_state = rnn_cell(x_embed[:, i, :], [cur_state])[0]
        #     states.append(cur_state)
        #
        #
        # stacked_states = tf.stack(states, axis=1)
        #
        # self.logits = tf.layers.dense(stacked_states,self.num_tags)

        embed_size = 100
        state_size = 70
        embeds = tf.get_variable('embeddings',[self.num_terms, embed_size])
        x_embed = tf.nn.embedding_lookup(embeds,self.x)

        forward_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
        backward_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)

        (forward_cell, backward_cell), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, x_embed, sequence_length=self.lengths, dtype=tf.float32)
        op = tf.concat([forward_cell, backward_cell], axis=-1)
        self.logits = tf.layers.dense(op, units=self.num_tags, activation=None)

    # TODO(student): You must implement this.
    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.
        
        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """

        logits = self.sess.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)


        #return numpy.zeros_like(terms)

    # TODO(student): You must implement this.
    def build_training(self):
        """Prepares the class for training.
        
        It is up to you how you implement this function, as long as train_on_batch
        works.
        
        Hint:
            - Lookup tf.contrib.seq2seq.sequence_loss 
            - tf.losses.get_total_loss() should return a valid tensor (without raising
                an exception). Equivalently, tf.losses.get_losses() should return a
                non-empty list.
        """

        bin_mat = self.lengths_vector_to_binary_matrix(self.lengths)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        loss = tf.contrib.seq2seq.sequence_loss(self.logits,self.targets,bin_mat)

        self.train_op = tf.contrib.training.create_train_op(loss,optimizer)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



        pass


    def feed_input_in_train(self,slice_tag,slice_x,slice_len,is_training,learning_rate):
        return {self.targets: slice_tag,
                self.x: slice_x,
                self.lengths: slice_len,
                self.is_training: is_training,
                self.learning_rate: learning_rate}

    def train_epoch(self, terms, tags, lengths, batch_size=25, learn_rate=0.01):
        """Performs updates on the model given training training data.
        
        This will be called with numpy arrays similar to the ones created in 
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size. Grader script will not pass this,
                but it is only here so that you can experiment with a "good batch size"
                from your main block.
            learn_rate: float for learning rate. Grader script will not pass this,
                but it is only here so that you can experiment with a "good learn rate"
                from your main block.

        Return:
            boolean. You should return True iff you want the training to continue. If
            you return False (or do not return anyhting) then training will stop after
            the first iteration!
        """
        # <-- Your implementation goes here.
        # Finally, make sure you uncomment the `return True` below.
        # return True

        indices = numpy.random.permutation(terms.shape[0])
        for si in range(0,terms.shape[0],batch_size):
            se = min(si + batch_size, terms.shape[0])
            slice_x = terms[indices[si:se]]
            slice_tag = tags[indices[si:se]]
            slice_len = lengths[indices[si:se]]
            self.sess.run(self.train_op,self.feed_input_in_train(slice_tag,slice_x,slice_len,True,learn_rate))
        return True

    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths):
        pass


def main():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    for j in range(10):
        model.train_epoch(train_terms, train_tags, train_lengths)
        print('Finished epoch %i. Evaluating ...' % (j+1))
        model.evaluate(test_terms, test_tags, test_lengths)


if __name__ == '__main__':
    main()

