from collections import OrderedDict
import numpy as np
from skimage.measure import compare_ssim as ssim


# TODO: Refactoring
def eval_synthetic(it, gen, data, tag='', batch_size = 128, sampler=None):
    metrics = OrderedDict()

    if sampler is not None:
        z = sampler(batch_size * 8)       # TODO: Originally 1024
        samples = gen(z)        # Feed z
    else:
        samples = gen(batch_size * 8)     # Generate n images TODO: Originally 1024

    # Simple metric for MoG (VEEGAN, https://arxiv.org/abs/1705.07761)
    if 'get_hq_ratio' in dir(data) and 'get_n_modes' in dir(data):
        metrics['hq_ratio'] = data.get_hq_ratio(samples) * 100.0
        metrics['modes_ratio'] = data.get_n_modes(samples) / float(data.n_modes) * 100.0

    print "{}({}) ".format(tag, it), ', '.join(['{}={:.2f}'.format(k, v) for k, v in metrics.iteritems()])

    return metrics


# TODO: Refactoring
# Simple & naive evaluation function
def eval_images_naive(it, gen, data, tag='', sampler=None):
    metrics = OrderedDict()

    if sampler is not None:
        z = sampler(128)
        samples = gen(z)        # Feed z
    else:
        samples = gen(128)      # Generate n images

    true_samples = data.validation.images
    true_labels = data.validation.labels if 'labels' in dir(data.validation) else None


    # Compute dist.
    dist_func = lambda a, b: np.linalg.norm((a - b).reshape((-1)), ord=2)

    # Distance: (generated samples) x (true samples)
    dist = np.array([[dist_func(x, x_true) for x_true in true_samples] for x in samples])

    best_matching_i_true = np.argmin(dist, axis=1)
    metrics['n_modes'] = len(np.unique(best_matching_i_true))
    metrics['ave_dist'] = np.average(np.min(dist, axis=1))


    # Check the labels (if exist)
    if true_labels is not None:
        label_cnts = np.sum(true_labels[best_matching_i_true], axis=0)
        metrics['n_labels'] = np.sum(label_cnts > 0)


    # Compute SSIM among top-k candidates (XXX: No supporting evidence for this approx.)
    k = 10
    top_k_matching_samples = np.argpartition(dist, k, axis=1)[:, :k]

    # Please refer to https://en.wikipedia.org/wiki/Structural_similarity
    # compare_ssim assumes (W, H, C) ordering
    sim_func = lambda a, b: ssim(a, b, multichannel=True, data_range=2.0)

    # Similarity: (generated samples) x (top-k candidates)
    sim = [[sim_func(samples[i], true_samples[i_true]) for i_true in i_topk] \
                                for i, i_topk in enumerate(top_k_matching_samples)]
    sim = np.array(sim)

    metrics['ave_sim'] = np.average(np.max(sim, axis=1))


    # TODO: Impl. IvOM
    
    # TODO: Impl. better metrics

    print "Eval({}) ".format(it), ', '.join(['{}={:.2f}'.format(k, v) for k, v in metrics.iteritems()])

    return metrics
