import torch
import torch.nn as nn
import torch.nn.functional as nnf
import pdb

class SingleScaleRNGF(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.eta = kwargs['eta'] if 'eta' in kwargs else 1.0

        # Define Sobel filters for gradient computation in 2D and 3D
        self.sobel_x_2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x_2d.weight = nn.Parameter(torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32), requires_grad=False)
        self.sobel_y_2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y_2d.weight = nn.Parameter(torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32), requires_grad=False)

        # For 3D, define additional filter for z direction
        self.sobel_x_3d = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x_3d.weight = nn.Parameter(torch.tensor([[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]], dtype=torch.float32), requires_grad=False)
        self.sobel_y_3d = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y_3d.weight = nn.Parameter(torch.tensor([[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-2, -4, -2], [0, 0, 0], [2, 4, 2]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]], dtype=torch.float32), requires_grad=False)
        self.sobel_z_3d = nn.Conv3d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_z_3d.weight = nn.Parameter(torch.tensor([[[[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]], dtype=torch.float32), requires_grad=False)

    def forward(self, pred, target, mask):
        fixed_grad = self.compute_normalized_gradient(target, mask).detach()
        warped_grad = self.compute_normalized_gradient(pred, mask)

        # return torch.mean((fixed_grad - warped_grad) ** 2)
        inner_product = torch.sum(fixed_grad * warped_grad, dim=1)
        loss = 1 - torch.mean(inner_product ** 2)
        return loss

    def compute_normalized_gradient(self, image, mask):
        # Assuming image is of shape [batch, channel, ..., ...]

        device = image.device
        # For 3D images, include gradient in z direction
        if image.dim() == 5:  # 5D tensor includes batch and channel dimensions
            self.sobel_x_3d.to(device)
            self.sobel_y_3d.to(device)
            self.sobel_z_3d.to(device)
            grad_x = self.sobel_x_3d(image)
            grad_y = self.sobel_y_3d(image)
            grad_z = self.sobel_z_3d(image)
            grad = torch.cat([grad_x, grad_y, grad_z], dim=1)

        else:  # For 2D images
            self.sobel_x_2d.to(device)
            self.sobel_y_2d.to(device)
            grad_x = self.sobel_x_2d(image)
            grad_y = self.sobel_y_2d(image)
            grad = torch.cat([grad_x, grad_y], dim=1)

        # Normalize the gradient vector
        eps = torch.finfo(image.dtype).eps
        norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + eps)
        mask = mask.squeeze(1) # remove the channel dimension
        if image.dim() == 5:
            dims = (1, 2, 3)
        else:
            dims = (1, 2)
        auto_eps = self.eta / torch.sum(mask, dim=dims) * torch.sum(norm*mask, dim=dims)
        auto_eps = auto_eps.view(-1, *[1 for _ in range(image.dim() - 2)])

        normalized_grad = grad / torch.sqrt(torch.sum(grad ** 2, dim=1) + auto_eps ** 2 + eps).unsqueeze(1)
        return normalized_grad

class RNGF(torch.nn.Module):
    """
    Multi-scale loss from C2FViT: https://github.com/cwmok/C2FViT
    """
    def  __init__(self, **kwargs):
        super().__init__()
        self.num_scales = kwargs['scale'] if 'scale' in kwargs else 1
        self.kernel = kwargs['kernel'] if 'kernel' in kwargs else 3
        self.half_resolution = kwargs['half_resolution'] if 'half_resolution' in kwargs else 0

        self.similarity_metric = []
        for i in range(self.num_scales):
            self.similarity_metric.append(
                        SingleScaleRNGF()
            )

    def forward(self, I, J, mask):
        dim = I.dim() - 2
        if self.half_resolution:
            kwargs = {'scale_factor':0.5, 'align_corners':True}
            if dim == 2:
                I = nnf.interpolate(I, mode='bilinear', **kwargs)
                J = nnf.interpolate(J, mode='bilinear', **kwargs)
                mask = nnf.interpolate(mask, mode='bilinear', **kwargs)
            elif dim == 3:
                I = nnf.interpolate(I, mode='trilinear', **kwargs)
                J = nnf.interpolate(J, mode='trilinear', **kwargs)
                mask = nnf.interpolate(mask, mode='trilinear', **kwargs)

        if dim == 2:
            pooling_fn = nnf.avg_pool2d
        elif dim == 3:
            pooling_fn = nnf.avg_pool3d

        total_loss = []
        for i in range(self.num_scales):
            current_loss = self.similarity_metric[i](I, J, mask)
            total_loss.append(current_loss / self.num_scales)

            I = pooling_fn(I, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            J = pooling_fn(J, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            mask = pooling_fn(mask, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)

        return sum(total_loss)
