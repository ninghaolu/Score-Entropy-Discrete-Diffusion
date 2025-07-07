import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]       # [B, 1] → [B]
        next_sigma = self.noise(t - step_size)[0]       # [B, 1] → [B]
        dsigma = curr_sigma - next_sigma        # [B]

        score = score_fn(x, curr_sigma)     # input: x [B, L], sigma [B], output: score: [B, L, V]

        stag_score = self.graph.staggered_score(score, dsigma)      # [B, L, V]
        probs = stag_score * self.graph.transp_transition(x, dsigma)    # [B, L, V]
        return sample_categorical(probs)     # [B, L]

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]        # [B, 1]

        score = score_fn(x, sigma)        # [B, L, V]
        stag_score = self.graph.staggered_score(score, sigma)       # [B, L, V]
        probs = stag_score * self.graph.transp_transition(x, sigma)         # [B, L, V]
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)        # [B, L]
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun            # for possible future usage, currently f(x) = x
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)      # [B, L], initialization, every position is [MASK]
        timesteps = torch.linspace(1, eps, steps + 1, device=device)        # divide [1 -> \epsilon] into "steps" steps
        dt = (1 - eps) / steps  # The step size for the reverse diffusion process.

        # Reverse diffusion loop: from t = 1 → eps
        for i in range(steps):
            # Construct the current time tensor for each sample in the batch.
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)   # shape: [B, 1]
            x = projector(x)    # shape: [B, L]
            # One reverse diffusion update step: x_t → x_{t-1}
            x = predictor.update_fn(sampling_score_fn, x, t, dt)    # shape: [B, L]
            
        # Optional final denoising step (produces sharper results).
        if denoise:
            # denoising step
            x = projector(x)
            # Use the final time step (eps) for denoising.
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t)     # [B, L]
        # Return the final sampled token indices 
        # Each token is an integer ID in [0, vocab_size)
        # Shape: [B, L]
        return x
    
    return pc_sampler

