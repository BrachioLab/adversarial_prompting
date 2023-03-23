"""
Base class for bayesian optimization objectives 
"""

class Objective:
    '''Base class for any optimization task
        class supports oracle calls and tracks
        the total number of oracle class made during 
        optimization 
    ''' 
    def __init__(
        self,
        num_calls=0,
        task_id='',
        dim=256,
        lb=None,
        ub=None,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks 
        self.task_id = task_id
        # latent dimension of vaae
        self.dim = dim
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub  

    def __call__(self, xs):
        ''' Input 
                x: a numpy array or pytorch tensor of search space points
            Output
                out_dict['valid_xs'] = an array of xs passed in from which valid scores were obtained 
                out_dict['scores']: an array of valid scores obtained from input xs
        '''
        self.num_calls += xs.shape[0]
        return self.query_oracle(xs)


    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        raise NotImplementedError("Must implement query_oracle() specific to desired optimization task")
