# ppgpr
from .base import DenseNetwork
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.variational import LMCVariationalStrategy
from gpytorch.kernels.additive_structure_kernel import AdditiveStructureKernel

# Multi-task Variational GP:
# https://docs.gpytorch.ai/en/v1.4.2/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, multi_task=False, 
                    num_tasks=1, num_latents=1):
        if multi_task:
            # Use a different set of inducing points for each task (just copy same points for each task)
            inducing_points = inducing_points.unsqueeze(0)
            inducing_points = inducing_points.repeat(num_latents,1,1)  # num_tasks x n_inducing x d   =  (2, 500, 256)
            # We have to mark the CholeskyVariationalDistribution as batch
            # so that we learn a variational distribution for each task 
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_latents]) )
            # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
            # so that the output will be a MultitaskMultivariateNormal rather than a batch output
            variational_strategy = LMCVariationalStrategy(VariationalStrategy(self, 
                    inducing_points, variational_distribution, learn_inducing_locations=True),
                    num_tasks=num_tasks,
                    num_latents=num_latents,
                    latent_dim=-1  )
        else:
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super(GPModel, self).__init__(variational_strategy)

        if multi_task:
            # The mean and covariance modules should be marked as batch
            # so we learn a different set of hyperparameters
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents])
                    )
        else:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.num_outputs = 1
        
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)


class GPModel_Additive_Kernel(ApproximateGP): # PPGPR w/ Additive Kernel, No DKL
    # ADDITIVE KERNEL SOURCE:
    # https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/kernels/additive_structure_kernel.py
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super(GPModel_Additive_Kernel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = AdditiveStructureKernel(base_kernel, inducing_points.shape[-1])
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)

class SpecializedAdditiveGP(ApproximateGP):
    def __init__(
        self, 
        inducing_points, 
        likelihood, 
        num_tokens,
        hidden_dims=(32, 32), 
    ): 
        mean_modules = []
        covar_modules = []
        feature_extractors = []
        inducing_points_list = []
        inducing_points = inducing_points.reshape(inducing_points.shape[0], num_tokens, 768)
        for token_num in range(num_tokens):
            feature_extractor = DenseNetwork(
                input_dim=768, 
                hidden_dims=hidden_dims).to(inducing_points.device)
            
            inducing_pointsi = inducing_points[:, token_num, :]  
            inducing_pointsi = feature_extractor(inducing_pointsi)
            inducing_points_list.append(inducing_pointsi)
            feature_extractors.append(feature_extractor)
            mean_modules.append(gpytorch.means.ConstantMean() )
            covar_modules.append(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) )
        
        inducing_points = torch.cat(inducing_points_list)
        # inducing_points = feature_extractors(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SpecializedAdditiveGP, self).__init__(variational_strategy)
        self.num_outputs = 1 
        self.likelihood = likelihood
        self.feature_extractors = feature_extractors
        self.mean_modules = mean_modules
        self.covar_modules = covar_modules
        self.num_tokens = num_tokens

    def forward(self, x):
        posteriors = [] 
        import pdb 
        pdb.set_trace() 
        x = x.reshape(x.shape[0], self.num_tokens, -1)
        for token_num in range(self.num_tokens):
            input = x[:, token_num, :]  
            # input = self.feature_extractors[token_num](input)
            mean_x = self.mean_modules[token_num](input)
            covar_x = self.covar_modules[token_num](input)
            posteriors.append(gpytorch.distributions.MultivariateNormal(mean_x, covar_x) )
        posterior = posteriors[0]
        for gp in posteriors[1:]:
            posterior = posterior + gp
        return posterior 

    def __call__(self, x, *args, **kwargs):
        x = x.reshape(x.shape[0], self.num_tokens, 768)
        compressed = [] 
        for token_num in range(self.num_tokens):
            input = x[:, token_num, :]  
            input = self.feature_extractors[token_num](input)
            compressed.append(input)
        x = torch.cat(compressed)
        # x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)






class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(32, 32), 
                        multi_task=False, num_tasks=1, num_latents=1 ):
        feature_extractor = DenseNetwork(input_dim=inducing_points.size(-1), hidden_dims=hidden_dims).to(inducing_points.device)
        inducing_points = feature_extractor(inducing_points)

        if multi_task:
            # Use a different set of inducing points for each task (just copy same points for each task)
            inducing_points = inducing_points.unsqueeze(0)
            inducing_points = inducing_points.repeat(num_latents,1,1)  # num_tasks x n_inducing x d   =  (2, 500, 256)
            # We have to mark the CholeskyVariationalDistribution as batch
            # so that we learn a variational distribution for each task
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_latents]) )
            # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
            # so that the output will be a MultitaskMultivariateNormal rather than a batch output
            variational_strategy = LMCVariationalStrategy(VariationalStrategy(self, 
                    inducing_points, variational_distribution, learn_inducing_locations=True),
                    num_tasks=num_tasks,
                    num_latents=num_latents,
                    latent_dim=-1  )
        else:
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)


        super(GPModelDKL, self).__init__(variational_strategy)

        if multi_task:
            # The mean and covariance modules should be marked as batch
            # so we learn a different set of hyperparameters 
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents])
                    )
        else:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.num_outputs = 1 #must be one

        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)



class GPModelSharedDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, shared_feature_extractor, 
                        multi_task=False, num_tasks=1, num_latents=1):
        # feature_extractor = DenseNetwork(input_dim=inducing_points.size(-1), hidden_dims=hidden_dims).to(inducing_points.device)
        inducing_points = shared_feature_extractor(inducing_points)

        if multi_task:
            # Use a different set of inducing points for each task (just copy same points for each task)
            inducing_points = inducing_points.unsqueeze(0)
            inducing_points = inducing_points.repeat(num_latents,1,1)  # num_tasks x n_inducing x d   =  (2, 500, 256)
            # We have to mark the CholeskyVariationalDistribution as batch
            # so that we learn a variational distribution for each task
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([num_latents]) )
            # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
            # so that the output will be a MultitaskMultivariateNormal rather than a batch output
            variational_strategy = LMCVariationalStrategy(VariationalStrategy(self, 
                    inducing_points, variational_distribution, learn_inducing_locations=True),
                    num_tasks=num_tasks,
                    num_latents=num_latents,
                    latent_dim=-1  )
        else:
            variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)


        super(GPModelSharedDKL, self).__init__(variational_strategy)

        if multi_task:
            # The mean and covariance modules should be marked as batch
            # so we learn a different set of hyperparameters
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
            self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),batch_shape=torch.Size([num_latents])
                    )
        else:
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.num_outputs = 1 #must be one 

        self.likelihood = likelihood
        self.feature_extractor = shared_feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)


class GPModelDKL_Additive_Kernel(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(256, 256)):
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            )
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModelDKL_Additive_Kernel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = AdditiveStructureKernel(base_kernel, inducing_points.shape[-1])
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            # self.model.eval() 
            self.likelihood.eval()
            dist = self.likelihood(self(X))

            return GPyTorchPosterior(mvn=dist)