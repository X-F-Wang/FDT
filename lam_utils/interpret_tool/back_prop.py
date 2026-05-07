import cv2
import numpy as np
import torch

from .utils import _add_batch_one, IG_baseline, interpolation, isotropic_gaussian_kernel

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        coord_x = -1+(2*i+1)/W
        coord_y = -1+(2*i+1)/H
        normalize to (-1, 1)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def attribution_objective(attr_func, h, w, window=16):
    def calculate_objective(image):
        return attr_func(image, h, w, window=window)

    return calculate_objective


def saliency_map_gradient(numpy_image, model, attr_func):
    img_tensor = torch.from_numpy(numpy_image)
    img_tensor.requires_grad_(True)
    result = model(_add_batch_one(img_tensor))
    target = attr_func(result)
    target.backward()
    return img_tensor.grad.numpy(), result


def I_gradient(numpy_image, baseline_image, model, attr_objective, fold, interp='linear'):
    interpolated = interpolation(numpy_image, baseline_image, fold, mode=interp).astype(np.float32)
    grad_list = np.zeros_like(interpolated, dtype=np.float32)
    result_list = []
    for i in range(fold):
        img_tensor = torch.from_numpy(interpolated[i])
        img_tensor.requires_grad_(True)
        result = model(_add_batch_one(img_tensor))
        target = attr_objective(result)
        target.backward()
        grad = img_tensor.grad.numpy()
        grad_list[i] = grad
        result_list.append(result)
    results_numpy = np.asarray(result_list)
    return grad_list, results_numpy, interpolated


def GaussianBlurPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        h, w, c = cv_numpy_image.shape
        kernel_interpolation = np.zeros((fold + 1, l, l))
        image_interpolation = np.zeros((fold, h, w, c))
        lambda_derivative_interpolation = np.zeros((fold, h, w, c))
        sigma_interpolation = np.linspace(sigma, 0, fold + 1)
        for i in range(fold + 1):
            kernel_interpolation[i] = isotropic_gaussian_kernel(l, sigma_interpolation[i])
        for i in range(fold):
            image_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, kernel_interpolation[i + 1])
            lambda_derivative_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, (
                    kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)

    return path_interpolation_func


def GaussianLinearPath(sigma, fold, l=5):
    def path_interpolation_func(cv_numpy_image):
        kernel = isotropic_gaussian_kernel(l, sigma)
        baseline_image = cv2.filter2D(cv_numpy_image, -1, kernel)
        image_interpolation = interpolation(cv_numpy_image, baseline_image, fold, mode='linear').astype(np.float32)
        lambda_derivative_interpolation = np.repeat(np.expand_dims(cv_numpy_image - baseline_image, axis=0), fold,
                                                    axis=0)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)

    return path_interpolation_func


def LinearPath(fold):
    def path_interpolation_func(cv_numpy_image):
        baseline_image = np.zeros_like(cv_numpy_image)
        image_interpolation = interpolation(cv_numpy_image, baseline_image, fold, mode='linear').astype(np.float32)
        lambda_derivative_interpolation = np.repeat(np.expand_dims(cv_numpy_image - baseline_image, axis=0), fold,
                                                    axis=0)
        return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
               np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)

    return path_interpolation_func


def Path_gradient(numpy_image, model, attr_objective, path_interpolation_func, cuda=False):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_numpy_images
    :return:
    """
    # if cuda:
    #     model = model.cuda()
    cv_numpy_image = np.moveaxis(numpy_image, 0, 2)
    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(cv_numpy_image)
    grad_accumulate_list = np.zeros_like(image_interpolation)
    result_list = []
    for i in range(image_interpolation.shape[0]):
        img_tensor = torch.from_numpy(image_interpolation[i])
        img_tensor.requires_grad_(True)
        inp_h, inp_w = img_tensor.shape[1], img_tensor.shape[2]
        coords = []
        scale_list = [4]

        for idx in range(len(scale_list)):
            h = round(inp_h * np.prod(scale_list[:idx + 1]))
            w = round(inp_w * np.prod(scale_list[:idx + 1]))

            coord = make_coord((h, w), flatten=False).unsqueeze(0)

            coords.append(coord)

        cell = torch.ones(2).unsqueeze(0)
        cell[:, 0] *= 2. / h
        cell[:, 1] *= 2. / w
        for idx in range(len(coords)):
            coords[idx] = coords[idx].cuda()

        cell = cell.cuda()

        data_norm = {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}
        t = data_norm['inp']
        inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
        t = data_norm['gt']
        gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1)
        gt_div = torch.FloatTensor(t['div']).view(1, 1, -1)
        inp = img_tensor.unsqueeze(0).cuda()

        if cuda:
            result = model.chop_forward((inp - inp_sub) / inp_div, coords, cell, 30000)
            result = result.permute(0, 2, 1).reshape(1, 3, h, w)
            target = attr_objective(result)
            target.backward()
            img_tensor=img_tensor.requires_grad_(True)
            grad = img_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
        else:
            result = model(_add_batch_one(img_tensor))
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

        grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i]
        result_list.append(result.cpu().detach())
    results_numpy = np.asarray(result_list, dtype=object)
    return grad_accumulate_list, results_numpy, image_interpolation


def saliency_map_PG(grad_list, result_list):
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]


def saliency_map_P_gradient(
        numpy_image, model, attr_objective, path_interpolation_func):
    grad_list, result_list, _ = Path_gradient(numpy_image, model, attr_objective, path_interpolation_func)
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]


def saliency_map_I_gradient(
        numpy_image, model, attr_objective, baseline='gaus', fold=10, interp='linear'):
    """
    :param numpy_image: RGB C x H x W
    :param model:
    :param attr_func:
    :param h:
    :param w:
    :param window:
    :param baseline:
    :return:
    """
    numpy_baseline = np.moveaxis(IG_baseline(np.moveaxis(numpy_image, 0, 2) * 255., mode=baseline) / 255., 2, 0)
    grad_list, result_list, _ = I_gradient(numpy_image, numpy_baseline, model, attr_objective, fold, interp='linear')
    final_grad = grad_list.mean(axis=0) * (numpy_image - numpy_baseline)
    return final_grad, result_list[-1]
