# adjusted from https://github.com/a1302z/ObjaxDPTraining/tree/main/dptraining
from typing import Callable, Union, Optional, Tuple, abstractmethod

from functools import partial
from objax import nn, functional, Module
from objax.constants import ConvPadding
from objax.util import local_kwargs
from objax.typing import JaxArray

from objax.functional import flatten

class ResNet9(Module):  
    def __init__(  
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        conv_cls: Module = nn.Conv2D,
        act_func: Module = functional.relu,
        norm_cls: Module = nn.GroupNorm2D,
        pool_func: Callable = partial(functional.max_pool_2d, size=2),
        linear_cls: Module = nn.Linear,
        out_func: Callable = lambda x: x,
        scale_norm: bool = False,
        num_groups: tuple[int, ...] = (32, 32, 32, 32),
        channels: tuple[int, ...] = (64, 128, 256, 256),
    ):
        """9-layer Residual resnet9_gnwork. Architecture:
        conv-conv-Residual(conv, conv)-conv-conv-Residual(conv-conv)-FC
        """
        super().__init__()

        assert len(num_groups) == 4, "num_groups must be a tuple with 4 members"
        groups = num_groups

        self.conv1 = conv_norm_act(
            in_channels,
            channels[0],
            conv_cls=conv_cls,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[0],
        )
        self.conv2 = conv_norm_act(
            channels[0],
            channels[1],
            conv_cls=conv_cls,
            pool_func=pool_func,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[0],
        )

        self.res1 = nn.Sequential(
            [
                conv_norm_act(
                    channels[1],
                    channels[1],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[1],
                ),
                conv_norm_act(
                    channels[1],
                    channels[1],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[1],
                ),
            ]
        )

        self.conv3 = conv_norm_act(
            channels[1],
            channels[2],
            conv_cls=conv_cls,
            pool_func=pool_func,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[2],
        )
        self.conv4 = conv_norm_act(
            channels[2],
            channels[3],
            conv_cls=conv_cls,
            pool_func=pool_func,
            act_func=act_func,
            norm_cls=norm_cls,
            num_groups=groups[2],
        )

        self.res2 = nn.Sequential(
            [
                conv_norm_act(
                    channels[3],
                    channels[3],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[3],
                ),
                conv_norm_act(
                    channels[3],
                    channels[3],
                    conv_cls=conv_cls,
                    act_func=act_func,
                    norm_cls=norm_cls,
                    num_groups=groups[3],
                ),
            ]
        )

        self.pooling = AdaptivePooling(functional.average_pool_2d, 2)
        self.flatten = Flatten()
        self.classifier = linear_cls(4 * channels[3], num_classes)

        if scale_norm:
            self.scale_norm_1 = (
                partial(norm_cls, groups=min(num_groups[1], channels[1]))
                if is_groupnorm(norm_cls)
                else norm_cls
            )(nin=channels[1])
            self.scale_norm_2 = (
                partial(norm_cls, groups=min(num_groups[3], channels[3]))
                if is_groupnorm(norm_cls)
                else norm_cls
            )(nin=channels[3])
        else:
            self.scale_norm_1 = lambda x: x
            self.scale_norm_2 = lambda x: x

        self.out_func = out_func

    def __call__(self, xb, *args, **kwargs):
        out = self.conv1(xb, *args, **local_kwargs(kwargs, self.conv1))
        out = self.conv2(out, *args, **local_kwargs(kwargs, self.conv2))
        out = self.res1(out, *args, **local_kwargs(kwargs, self.res1)) + out
        out = self.scale_norm_1(out, *args, **local_kwargs(kwargs, self.scale_norm_1))
        out = self.conv3(out, *args, **local_kwargs(kwargs, self.conv3))
        out = self.conv4(out, *args, **local_kwargs(kwargs, self.conv4))
        out = self.res2(out, *args, **local_kwargs(kwargs, self.res2)) + out
        out = self.scale_norm_2(out, *args, **local_kwargs(kwargs, self.scale_norm_2))
        out = self.pooling(out, *args, **local_kwargs(kwargs, self.pooling))
        out = self.flatten(out)
        out = self.classifier(out, *args, **local_kwargs(kwargs, self.classifier))
        out = self.out_func(out)
        return out

    def feature_extractor(self, xb, *args, **kwargs):
        out = self.conv1(xb, *args, **local_kwargs(kwargs, self.conv1))
        out = self.conv2(out, *args, **local_kwargs(kwargs, self.conv2))
        out = self.res1(out, *args, **local_kwargs(kwargs, self.res1)) + out
        out = self.scale_norm_1(out, *args, **local_kwargs(kwargs, self.scale_norm_1))
        out = self.conv3(out, *args, **local_kwargs(kwargs, self.conv3))
        out = self.conv4(out, *args, **local_kwargs(kwargs, self.conv4))
        out = self.res2(out, *args, **local_kwargs(kwargs, self.res2)) + out
        out = self.scale_norm_2(out, *args, **local_kwargs(kwargs, self.scale_norm_2))
        out = self.pooling(out, *args, **local_kwargs(kwargs, self.pooling))
        out = self.flatten(out)
        return out



class AdaptivePooling(Module):
    def __init__(
        self,
        pool_func: Callable,
        output_size: Union[int, Tuple[int, int]],
        stride: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pool_func = pool_func
        if isinstance(output_size, int):
            self.output_size_x = output_size
            self.output_size_y = output_size
        elif isinstance(output_size, tuple):
            self.output_size_x, self.output_size_y = output_size
        else:
            raise ValueError("output size must be either int or tuple of ints")
        self.stride = stride

    @staticmethod
    def _calc_kernel(inpt_size, outpt_size, stride) -> int:
        return inpt_size - (outpt_size - 1) * stride

    def __call__(self, x):
        _, _, size_x, size_y = x.shape
        if self.stride is None:
            stride = (size_x // self.output_size_x, size_y // self.output_size_y)
        else:
            stride = (self.stride, self.stride)
        k_x = AdaptivePooling._calc_kernel(size_x, self.output_size_x, stride[0])
        k_y = AdaptivePooling._calc_kernel(size_y, self.output_size_y, stride[1])
        return partial(self.pool_func, size=(k_x, k_y), strides=stride)(x)


def is_groupnorm(instance):
    return issubclass(instance, (nn.GroupNorm2D, ComplexGroupNormWhitening))


class Flatten(Module):
    def __call__(self, x: JaxArray) -> JaxArray:  # pylint:disable=arguments-differ
        return flatten(x)
    

def conv_norm_act(  
    in_channels,
    out_channels,
    conv_cls=nn.Conv2D,
    act_func=functional.relu,
    pool_func=lambda x: x,
    norm_cls=nn.GroupNorm2D,
    num_groups=32,
    ):
    if is_groupnorm(norm_cls):
        norm_cls = partial(norm_cls, groups=min(num_groups, out_channels))
    layers = [
        conv_cls(
            in_channels, out_channels, k=3, padding=ConvPadding.SAME, use_bias=False
        ),
        norm_cls(nin=out_channels),
        act_func,
    ]
    layers.append(pool_func)
    return nn.Sequential(layers)


class ComplexGroupNormWhitening(Module):
    """This exists for backwards compatibility reasons"""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass