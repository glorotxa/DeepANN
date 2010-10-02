"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA. 
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting 
 latent representation y is then mapped back to a "reconstructed" vector 
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight 
 matrix W' can optionally be constrained such that W' = W^T, in which case 
 the autoencoder is said to have tied weights. The network is trained such 
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into 
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means 
 of a stochastic mapping. Afterwards y is computed as before (using 
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction 
 error is now measured between z and the uncorrupted input x, which is 
 computed as the cross-entropy : 
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and 
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing 
   Systems 19, 2007

"""

import numpy, time, cPickle, gzip, sys, os

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg  import MRG_RandomStreams as RandomStreams


import PIL.Image


class dA(object):
    """Denoising Auto-Encoder class (dA) 

    A denoising autoencoders tries to reconstruct the input from a corrupted 
    version of it by projecting it first in a latent space and reprojecting 
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2) 
    computes the projection of the input into the latent space. Equation (3) 
    computes the reconstruction of the input, while equation (4) computes the 
    reconstruction error.
  
    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(self, numpy_rng, theano_rng = None, input = None, n_visible= 784, n_hidden= 500, 
               act = 'rect', noise = 'binomial', W = None, W_prime = None, bhid = None, bvis = None):
        """
        Initialize the dA class by specifying the number of visible units (the 
        dimension d of the input ), the number of hidden units ( the dimension 
        d' of the latent or hidden space ) and the corruption level. The 
        constructor also receives symbolic variables for the input, weights and 
        bias. Such a symbolic variables are useful when, for example the input is 
        the result of some computations, or when weights are shared between the 
        dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1, 
        and the weights of the dA are used in the second stage of training 
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is generated
                     based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for standalone
                      dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be 
                  shared belong the dA and another architecture; if dA should 
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for 
                     hidden units) that should be shared belong dA and another 
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for 
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.act = act
        self.noise = noise
        self.n_visible = n_visible
        self.n_hidden  = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng : 
            theano_rng = RandomStreams(rng.randint(2**30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -sqrt(6./(n_visible+n_hidden)) and
            # sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray( numpy_rng.uniform( 
                      low  = -numpy.sqrt(6./(n_hidden+n_visible)), 
                      high =  numpy.sqrt(6./(n_hidden+n_visible)), 
                      size = (n_visible, n_hidden)), dtype = theano.config.floatX)
            W = theano.shared(value = initial_W, name ='W')

        if not W_prime:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -sqrt(6./(n_visible+n_hidden)) and
            # sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W_prime = numpy.asarray( numpy_rng.uniform( 
                      low  = -numpy.sqrt(6./(n_hidden+n_visible)), 
                      high =  numpy.sqrt(6./(n_hidden+n_visible)), 
                      size = (n_hidden, n_visible)), dtype = theano.config.floatX)
            W_prime = theano.shared(value = initial_W_prime, name ='W_prime')

        if not bvis:
            bvis = theano.shared(value = numpy.zeros(n_visible, 
                                         dtype = theano.config.floatX))

        if not bhid:
            bhid = theano.shared(value = numpy.zeros(n_hidden,
                                dtype = theano.config.floatX), name ='b')


        self.W = W
        # b corresponds to the bias of the hidden 
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # un - tied weights
        self.W_prime = W_prime 
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None : 
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.matrix(name = 'input') 
        else:
            self.x = input

        self.params = [self.W, self.W_prime, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """ This function keeps ``1-corruption_level`` entries of the inputs the same 
        and zero-out randomly selected subset of size ``coruption_level`` 
        Note : first argument of theano.rng.binomial is the shape(size) of 
               random numbers that it should produce
               second argument is the number of trials 
               third argument is the probability of success of any trial
        
                this will produce an array of 0s and 1s where 1 has a probability of 
                1 - ``corruption_level`` and 0 with ``corruption_level``

                The binomial function return int64 data type by default. 
                int64 multiplicated by the input type(floatX) always return float64.
                To keep all data in floatX when floatX is float32, we set the dtype
                of the binomial to floatX. As in our case the value of the binomial 
                is always 0 or 1, this don't change the result. This is needed to allow
                the gpu to work correctly as it only support float32 for now.
        """
        if self.noise == 'binomial':
            return self.theano_rng.binomial( size = input.shape, n = 1, p =  1 - corruption_level, dtype=theano.config.floatX) * input
        if self.noise == 'binomial_NLP':
            return self.theano_rng.binomial( size = input.shape, n = 1, p =  1 - corruption_level[0], dtype=theano.config.floatX) * input \
                    + (input==0) * self.theano_rng.binomial( size = input.shape, n = 1, p =  corruption_level[1], dtype=theano.config.floatX)
        if self.noise == 'gaussian': # same as binomial_NLP
            return self.theano_rng.normal( size = input.shape, std = corruption_level, dtype=theano.config.floatX) + input

    
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        if self.act == 'sigmoid':
            return  T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        if self.act == 'tanh':
            return T.tanh(T.dot(input, self.W) + self.b)
        if self.act == 'rectifier':
            return ((T.dot(input, self.W) + self.b) >= 0) * (T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden ):
        """ Computes the reconstructed input given the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    
    def get_cost_updates(self, corruption_level, learning_rate, l2reg=0., l1reg=0.):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        if corruption_level == None:
            tilde_x = self.x
        else:
            tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y       = self.get_hidden_values( tilde_x)
        z       = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using minibatches,
        #        L will  be a vector, with one entry per example in minibatch
        
        XE = self.x * T.log(z) + (1 - self.x) *  T.log(1-z)
        cost = -T.mean(T.sum(XE, axis=1),axis=0)
        
        if l2reg != 0.:
            cost += l2reg * (T.mean(T.sum(self.W*self.W,1),0) + T.mean(T.sum(self.W_prime*self.W_prime,1),0))
        if l1reg != 0.:
            cost += l1reg * (T.mean(T.sum(T.abs_(y),1),0) + T.mean(T.sum(T.abs_(y),1),0))
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters 
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param -  learning_rate*gparam
    
        return (cost, updates)


########################################################## utils


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############
    print '... loading data'

    # Load the dataset 
    f = gzip.open(dataset,'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()


    def shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables
        
        The reason we store our dataset in shared variables is to allow 
        Theano to copy it into the GPU memory (when code is run on GPU). 
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared 
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are 
        # floats it doesn't make sense) therefore instead of returning 
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval




def tile_raster_images(X, img_shape, tile_shape,tile_spacing = (0,0), 
              scale_rows_to_unit_interval = True, output_pixel_vals = True):
    """
    Transform an array with one flattened image per row, into an array in 
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows 
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can 
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.  
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """
 
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as 
    # follows : 
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp 
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image 
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct 
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:,:,i] = numpy.zeros(out_shape,
                        dtype=dt)+channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it 
                # in the output
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel 
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)


        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1 
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the 
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H+Hs):tile_row*(H+Hs)+H,
                        tile_col * (W+Ws):tile_col*(W+Ws)+W
                        ] \
                        = this_img * c
        return out_array


####################################################################


def test_dA( learning_rate = 0.01, training_epochs = 15, dataset ='/u/glorotxa/mnist.pkl.gz',
        batch_size = 10, output_folder = 'dA_plots' ):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training 

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.value.shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch 
    x     = T.matrix('x')  # the data is presented as rasterized images

    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng        = numpy.random.RandomState(123)
    theano_rng = RandomStreams( rng.randint(2**30))

    da = dA(numpy_rng = rng, theano_rng = theano_rng, input = x,
            n_visible = 28*28, n_hidden = 500,act = 'rect', noise = 'binomial')

    cost, updates = da.get_cost_updates(corruption_level = 0.,
                                learning_rate = learning_rate)

    
    train_da = theano.function([index], cost, updates = updates,
         givens = {x:train_set_x[index*batch_size:(index+1)*batch_size]})
    
    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost '%epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % ((training_time)/60.))
    image = PIL.Image.fromarray(tile_raster_images( X = da.W.value.T,
                 img_shape = (28,28),tile_shape = (10,10), 
                 tile_spacing=(1,1)))
    image.save('filters_corruption_0.png') 
 
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng        = numpy.random.RandomState(123)
    theano_rng = RandomStreams( rng.randint(2**30))

    da = dA(numpy_rng = rng, theano_rng = theano_rng, input = x,
            n_visible = 28*28, n_hidden = 500)

    cost, updates = da.get_cost_updates(corruption_level = 0.3,
                                learning_rate = learning_rate)

    
    train_da = theano.function([index], cost, updates = updates,
         givens = {x:train_set_x[index*batch_size:(index+1)*batch_size]})

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost '%epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file '+os.path.split(__file__)[1]+' ran for %.2fm' % (training_time/60.))

    image = PIL.Image.fromarray(tile_raster_images( X = da.W.value.T,
                 img_shape = (28,28),tile_shape = (10,10), 
                 tile_spacing=(1,1)))
    image.save('filters_corruption_30.png') 
 
    os.chdir('../')


if __name__ == '__main__':
    test_dA()
