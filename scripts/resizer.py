# This code was taken from: https://github.com/assafshocher/resizer by Assaf Shocher
import numpy as np
import torch
from math import pi
from torch import nn
from guided_diffusion import dist_util
# device = torch.device('cuda:4')
device = dist_util.dev()
class Resizer(nn.Module):
    def __init__(self, in_shape, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
        super(Resizer, self).__init__()

        # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
        scale_factor, output_shape = self.fix_scale_and_size(in_shape, output_shape, scale_factor)

        # Choose interpolation method, each method has the matching kernel size
        method, kernel_width = {
            "cubic": (cubic, 4.0),
            "lanczos2": (lanczos2, 4.0),
            "lanczos3": (lanczos3, 6.0),
            "box": (box, 1.0),
            "linear": (linear, 2.0),
            None: (cubic, 4.0)  # set default interpolation method as cubic
            # None: (box, 1.0)  # set default interpolation method as cubic
        }.get(kernel)

        # Antialiasing is only used when downscaling
        antialiasing *= (np.any(np.array(scale_factor) < 1))
        # 如果antialiasing为True且缩放因子中有任何一个小于1的维度，将antialiasing置为True，否则为False。

        # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
        sorted_dims = np.argsort(np.array(scale_factor))
        # 根据缩放因子的大小对维度进行排序，得到按缩放因子从小到大排列的维度索引列表
        self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

        # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
        field_of_view_list = []
        weights_list = []
        for dim in self.sorted_dims:
            # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
            # weights that multiply the values there to get its result.
            print(f'inshape: {in_shape}; output_shape:{output_shape}. scale_factor:{scale_factor}, method: {method}, '
                  f'kernel_width:{kernel_width}')
            # inshape: (8, 3, 128, 128); output_shape:[8 3 8 8]. scale_factor:[1, 1, 0.0625, 0.0625]
            # inshape: (8, 3, 8, 8); output_shape:[  8   3 128 128]. scale_factor:[1, 1, 16, 16]
            weights, field_of_view = self.contributions(in_shape[dim], output_shape[dim], scale_factor[dim], method,
                                                        kernel_width, antialiasing)

            # convert to torch tensor
            weights = torch.tensor(weights.T, dtype=torch.float32)

            # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
            # tmp_im[field_of_view.T], (bsxfun style)
            weights_list.append(
                nn.Parameter(torch.reshape(weights, list(weights.shape) + (len(scale_factor) - 1) * [1]),
                             requires_grad=False))
            field_of_view_list.append(
                nn.Parameter(torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long), requires_grad=False))

        self.field_of_view = nn.ParameterList(field_of_view_list)
        self.weights = nn.ParameterList(weights_list)

    def forward(self, in_tensor):
        x = in_tensor

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):

            # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
            x = torch.transpose(x, dim, 0)

            # This is a bit of a complicated multiplication: x[field_of_view.T] is a tensor of order image_dims+1.
            # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
            # only, this is why it only adds 1 dim to 5the shape). We then multiply, for each pixel, its set of positions with
            # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
            # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
            # same number

            x = torch.sum(x[fov] * w.to(device), dim=0)

            # Finally we swap back the axes to the original order
            x = torch.transpose(x, dim, 0)

        return x

    def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
        # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
        # same size as the number of input dimensions)
        # 这段代码的作用是为了确保输入形状、输出形状和缩放因子的值满足特定的要求，并提供了一些默认的处理方式来处理缺失或未指定的参数。
        # 这样可以在后续的代码中使用修正后的值来进行相应的处理。

        # 如果提供了缩放因子scale_factor，则将其转化为标准化形式，即将其转换为与输入维度相同的缩放因子列表。
        # 如果scale_factor是一个标量且输入形状的维度大于1，则将其复制为一个具有两个相同值的列表。
        # 如果scale_factor的大小小于输入形状的维度，则通过将未指定的缩放因子设置为1来扩展scale_factor列表的大小，以使其与输入形状的大小相同。
        if scale_factor is not None:
            # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
            if np.isscalar(scale_factor) and len(input_shape) > 1:
                scale_factor = [scale_factor, scale_factor]

            # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
            scale_factor = list(scale_factor)
            scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor

        # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
        # to all the unspecified dimensions
        if output_shape is not None:
            output_shape = list(input_shape[len(output_shape):]) + list(np.uint(np.array(output_shape)))
        # 如果提供了输出形状output_shape，则将其扩展到与输入形状的大小相同。这是通过将未指定的维度设置为原始输入维度的大小，
        # 并将output_shape的类型转换为无符号整数来实现的。
        # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
        # sub-optimal, because there can be different scales to the same output-shape.
        # 如果没有提供缩放因子scale_factor，则根据output_shape计算缩放因子。这是通过计算输入形状与scale_factor的乘积得到的。
        if scale_factor is None:
            scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)
        # 如果没有提供输出形状output_shape，则根据缩放因子scale_factor计算输出形状。这是通过计算输入形状与scale_factor的乘积，
        # 并将结果向上取整后转换为无符号整数得到的。
        # Dealing with missing output-shape. calculating according to scale-factor
        if output_shape is None:
            output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

        return scale_factor, output_shape

    def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
        # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
        # such that each position from the field_of_view will be multiplied with a matching filter from the
        # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
        # around it. This is only done for one dimension of the image.
        # 该函数计算一组滤波器和视野范围，以便在后续应用中，每个视野范围的位置都与来自'weights'的匹配滤波器进行相乘
        # 这是仅针对图像的一个维度进行的

        # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
        # 1/sf. this means filtering is more 'low-pass filter'.
        # 当启用反锯齿时（默认仅用于降采样），接收域将扩展到大小为1/sf。这意味着滤波更像是'低通滤波器'
        fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
        kernel_width *= 1.0 / scale if antialiasing else 1.0

        # These are the coordinates of the output image
        out_coordinates = np.arange(1, out_length + 1)

        # since both scale-factor and output size can be provided simulatneously, perserving the center of the image requires shifting
        # the output coordinates. the deviation is because out_length doesn't necesary equal in_length*scale.
        # to keep the center we need to subtract half of this deivation so that we get equal margins for boths sides and center is preserved.
        # 为了保持图像中心位置不变，需要对输出坐标进行偏移
        shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2

        # These are the matching positions of the output-coordinates on the input image coordinates.
        # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
        # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
        # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
        # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
        # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
        # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
        # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
        # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
        # 输出坐标在输入图像坐标上的匹配位置
        match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)

        # 开始滤波器乘法的左边界，它取决于滤波器的大小
        # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
        # np.floor 下取整
        left_boundary = np.floor(match_coordinates - kernel_width / 2)

        # 由于边界具有子像素，核宽度需要进行扩展，以便它能够"看到"只涵盖了一部分像素的像素中心。因此，在两侧各添加一个像素（权重可以将其归零）
        # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
        # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
        # np.ceil 上取整
        expanded_kernel_width = np.ceil(kernel_width) + 2

        # Determine a set of field_of_view for each each output position, these are the pixels in the input image
        # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
        # vertical dim is the pixels it 'sees' (kernel_size + 2)
        # 为每个输出位置确定一个视野范围，即输入图像中的像素
        field_of_view = np.squeeze(
            np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1))

        # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
        # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
        # 'field_of_view')
        # 为视野范围内的每个像素分配权重
        weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

        # 将权重归一化，使其总和为1，避免除以0的情况
        # Normalize weights to sum up to 1. be careful from dividing by 0
        sum_weights = np.sum(weights, axis=1)
        sum_weights[sum_weights == 0] = 1.0
        weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

        # We use this mirror structure as a trick for reflection padding at the boundaries
        mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
        field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

        # Get rid of  weights and pixel positions that are of zero weight
        non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
        weights = np.squeeze(weights[:, non_zero_out_pixels])
        field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

        # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
        return weights, field_of_view


# These next functions are all interpolation methods. x is the distance from the left pixel center


def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) +
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def lanczos2(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (((np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps) /
             ((pi ** 2 * x ** 2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))