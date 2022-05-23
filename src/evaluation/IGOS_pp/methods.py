"""
Different explanation methods from the Integrated-Gradient Optimized Saliency map methods.
© copyright Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

from torch.autograd import Variable
from methods_helper import *


def IGOS(
        model,
        images,
        baselines,
        labels,
        L1=1,
        L2=20,
        size=28,
        ig_iter=15,
        iterations=20,
        alpha=8,
        softmax=True,
        **kwargs
):

    """
        Generates explanation by optimizing a mask with integrated gradient.
        Paper title:  Visualizing Deep Networks by Optimizing with Integrated Gradients, AAAI 2020
        Link to the paper: https://aaai.org/ojs/index.php/AAAI/article/view/6863/6717

    :param model:
    :param images:
    :param baselines:
    :param labels:
    :param L1:
    :param L2:
    :param size:
    :param ig_iter:
    :param iterations:
    :param alpha:
    :param softmax:
    :param kwargs:
    :return:
    """

    def regularization_loss(masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1) + \
               L2 * tv_norm(masks)

        # Define loss function for regularization terms

    def loss_function(up_masks, masks, indices, noise=True):
        losses = interval_score(
            model,
            images[indices],
            baselines[indices],
            labels[indices],
            up_masks,
            ig_iter,
            output_func,
            noise
            )
        return losses.sum(dim=1).view(-1) + regularization_loss(masks)

    # Create initial masks
    masks = torch.ones((images.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks = Variable(masks, requires_grad=True)

    if softmax:
        output_func = softmax_output
    else:
        logit_output.original = torch.gather(torch.nn.Sigmoid()(model(images)), 1, labels.view(-1, 1))
        output_func = logit_output

    for i in range(iterations):

        up_masks = upscale(masks)

        losses = regularization_loss(masks)
        losses.sum().backward()
        total_grads = masks.grad.clone()
        masks.grad.zero_()

        # Computer the integrated gradient
        integrated_gradient(model, images, baselines, labels, up_masks, ig_iter, output_func)
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        # Compute step
        alphas = line_search(masks, total_grads, loss_function, alpha)

        # Update the mask
        masks.data -= total_grads * alphas
        masks.grad.zero_()
        masks.data.clamp_(0, 1)

    return masks


def iGOS_pp(
        model,
        images,
        baselines,
        labels,
        size=28,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=20,
        alpha=1000,
        softmax=True,
        **kwargs):
    """
        Generates explanation by optimizing a separate masks for insertion and deletion.
        Paper title:  iGOS++: Integrated Gradient Optimized Saliency by Bilateral Perturbations
        Link to the paper: https://arxiv.org/pdf/2012.15783.pdf

    :param model:
    :param images:
    :param baselines:
    :param labels:
    :param size:
    :param iterations:
    :param ig_iter:
    :param L1:
    :param L2:
    :param alpha:
    :param softmax:
    :param kwargs:
    :return:
    """
    def regularization_loss(images, masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1) + \
               L2 * bilateral_tv_norm(images, masks, tv_beta=2, sigma=0.01)

    def ins_loss_function(up_masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    baselines[indices],
                    images[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise
                    )
        return losses.sum(dim=1).view(-1)

    def del_loss_function(up_masks, indices, noise=True):
        losses = interval_score(
                    model,
                    images[indices],
                    baselines[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise,
                    )
        return losses.sum(dim=1).view(-1)

    def loss_function(up_masks, masks, indices):
        loss = del_loss_function(up_masks[:, 0], indices)
        loss += ins_loss_function(up_masks[:, 1], indices)
        loss += del_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        loss += ins_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        return loss + regularization_loss(images[indices], masks[:, 0] * masks[:, 1])

    masks_del = torch.ones((images.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks_del = Variable(masks_del, requires_grad=True)
    masks_ins = torch.ones((images.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks_ins = Variable(masks_ins, requires_grad=True)

    if softmax:
        output_func = softmax_output
    else:
        logit_output.original = torch.gather(torch.nn.Sigmoid()(model(images)), 1, labels.view(-1,1))
        output_func = logit_output

    for i in range(iterations):
        up_masks1 = upscale(masks_del)
        up_masks2 = upscale(masks_ins)

        # Compute the integrated gradient for the combined mask, optimized for deletion
        integrated_gradient(model, images, baselines, labels, up_masks1 * up_masks2, ig_iter, output_func)
        total_grads1 = masks_del.grad.clone()
        total_grads2 = masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the combined mask, optimized for insertion
        integrated_gradient(model, baselines, images, labels, up_masks1 * up_masks2, ig_iter, output_func)
        total_grads1 -= masks_del.grad.clone()  # Negative because insertion loss is 1 - score.
        total_grads2 -= masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the deletion mask
        integrated_gradient(model, images, baselines, labels, up_masks1, ig_iter, output_func)
        total_grads1 += masks_del.grad.clone()
        masks_del.grad.zero_()

        # Compute the integrated graident for the insertion mask
        integrated_gradient(model, baselines, images, labels, up_masks2, ig_iter, output_func)
        total_grads2 -= masks_ins.grad.clone()
        masks_ins.grad.zero_()

        # Average them to balance out the terms with the regularization terms
        total_grads1 /= 2
        total_grads2 /= 2

        # Computer regularization for combined masks
        losses = regularization_loss(images, masks_del * masks_ins)
        losses.sum().backward()
        total_grads1 += masks_del.grad.clone()
        total_grads2 += masks_ins.grad.clone()

        masks = torch.cat((masks_del.unsqueeze(1), masks_ins.unsqueeze(1)), 1)
        total_grads = torch.cat((total_grads1.unsqueeze(1), total_grads2.unsqueeze(1)), 1)
        alphas = line_search(masks, total_grads, loss_function, alpha)

        masks_del.data -= total_grads1 * alphas
        masks_ins.data -= total_grads2 * alphas
        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0,1)
        masks_ins.data.clamp_(0,1)

    return masks_del * masks_ins


def iGOS_p(
        model,
        images,
        baselines,
        labels,
        size=28,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=20,
        alpha=1000,
        softmax=True,
        **kwargs):
    """
        Similar idea to iGOS++, but generates explanation only using one mask (optimized for both insertion and deletion).

    :param model:
    :param images:
    :param baselines:
    :param labels:
    :param size:
    :param iterations:
    :param ig_iter:
    :param L1:
    :param L2:
    :param alpha:
    :param softmax:
    :param kwargs:
    :return:
    """
    def regularization_loss(masks):
        return L1 * torch.mean(torch.abs(1-masks).view(masks.shape[0],-1), dim=1) +\
               L2 * bilateral_tv_norm(images, masks, tv_beta=2, sigma=0.01)

    def loss_function(up_masks, masks, indices, noise=True):
        losses = -interval_score(
                    model,
                    baselines[indices],
                    images[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise
                    )
        losses += interval_score(
                    model,
                    images[indices],
                    baselines[indices],
                    labels[indices],
                    up_masks,
                    ig_iter,
                    output_func,
                    noise
                    )
        return losses.sum(dim=1).view(-1) + regularization_loss(masks)

    masks = torch.ones((images.shape[0],1,size,size), dtype=torch.float32, device='cuda')
    masks = Variable(masks, requires_grad=True)

    if softmax:
        output_func = softmax_output
    else:
        logit_output.original = torch.gather(torch.nn.Sigmoid()(model(images)), 1, labels.view(-1,1))
        output_func = logit_output

    for i in range(iterations):
        total_grads = torch.zeros(images.shape[0], 1, size, size, dtype=torch.float32).cuda()
        up_masks = upscale(masks)

        integrated_gradient(model, images, baselines, labels, up_masks, ig_iter, output_func)
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        integrated_gradient(model, baselines, images, labels, up_masks, ig_iter, output_func)
        total_grads += -masks.grad.clone()
        masks.grad.zero_()

        losses = regularization_loss(masks)
        losses.sum().backward()
        total_grads += masks.grad.clone()
        masks.grad.zero_()

        alphas = line_search(masks, total_grads, loss_function, alpha)

        masks.data -= total_grads * alphas
        masks.grad.zero_()
        masks.data.clamp_(0, 1)

    return masks
