import torch

from samitorch.factories.layers import ActivationFunctionsFactory, NormalizationLayerFactory
from samitorch.configs.model_configurations import ModelConfiguration


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with zero padding

    Args:
        in_planes (int): The number of input channels.
        out_planes (int): The number of output channels.
        stride (int): The convolution stride. Controls the stride for the cross-correlation
        groups (int): The number of groups in the convolution. Controls the connections between input and outputs.
        dilation (int): Controls the spacing between the kernel points

    Returns:
        :obj:`torch.nn.Conv3d`: A convolution layer.
    """
    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution without padding.

    Args:
        in_planes (int): The number of input channels.
        out_planes (int): The number of output channels.
        stride (int): The convolution stride. Controls the stride for the cross-correlation

    Returns:
        :obj:`torch.nn.Conv3d`: A convolution layer.
    """

    return torch.nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(torch.nn.Module):
    """
    A basic ResNet block.
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, activation="ReLU", norm_num_groups=8):
        """
        Basic block initializer.
        Args:
            in_planes (int): The number of input channels.
            out_planes (int): The number of output channels.
            stride (int): The convolution stride. Controls the stride for the cross-correlation
            downsample (:obj:`torch.nn.Sequential`): A sequential downsampling sub-layer.
            groups (int): The number of groups in the convolution. Controls the connections between input and outputs.
            base_width (int): The base number of output channels.
            dilation (int): Controls the spacing between the kernel points
            activation (str): Desired non-linear activation function.
            norm_num_groups (int): The number of groups for group normalization.
        """
        super(BasicBlock, self).__init__()
        self._activation_layer_factory = ActivationFunctionsFactory()
        self._normalization_layer_factory = NormalizationLayerFactory()

        if norm_num_groups is not None:
            self.norm1 = self._normalization_layer_factory.get_layer("GroupNorm", norm_num_groups, out_planes)
            self.norm2 = self._normalization_layer_factory.get_layer("GroupNorm", norm_num_groups, out_planes)
        else:
            self.norm1 = self._normalization_layer_factory.get_layer("BatchNorm3d", out_planes)
            self.norm2 = self._normalization_layer_factory.get_layer("BatchNorm3d", out_planes)

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, out_planes, stride)

        if activation == "PReLU":
            self.activation = self._activation_layer_factory.get_layer(activation)
        else:
            self.activation = self._activation_layer_factory.get_layer(activation, True)

        self.conv2 = conv3x3(out_planes, out_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(torch.nn.Module):
    """
    Bottleneck ResNet block.
    """
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, activation="ReLU", norm_num_groups=8):
        """
        Bottleneck Block initializer.

        Args:
            in_planes (int): The number of input channels.
            out_planes (int): The number of output channels.
            stride (int): The convolution stride. Controls the stride for the cross-correlation
            downsample (:obj:`torch.nn.Sequential`): A sequential downsampling sub-layer.
            groups (int): The number of groups in the convolution. Controls the connections between input and outputs.
            base_width (int): The base number of output channels.
            dilation (int): Controls the spacing between the kernel points
            activation (str): Desired non-linear activation function.
            norm_num_groups (int): The number of groups for group normalization.
        """
        super(Bottleneck, self).__init__()

        self._activation_layer_factory = ActivationFunctionsFactory()
        self._normalization_layer_factory = NormalizationLayerFactory()

        width = int(out_planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if norm_num_groups is not None:
            self.norm1 = self._normalization_layer_factory.get_layer("GroupNorm", norm_num_groups, width)
            self.norm2 = self._normalization_layer_factory.get_layer("GroupNorm", norm_num_groups, width)
            self.norm3 = self._normalization_layer_factory.get_layer("GroupNorm", norm_num_groups,
                                                                     out_planes * self.expansion)
        else:
            self.norm1 = self._normalization_layer_factory.get_layer("BatchNorm3d", width)
            self.norm2 = self._normalization_layer_factory.get_layer("BatchNorm3d", width)
            self.norm3 = self._normalization_layer_factory.get_layer("BatchNorm3d", out_planes * self.expansion)

        self.conv1 = conv1x1(in_planes, width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.conv3 = conv1x1(width, out_planes * self.expansion)

        if activation == "PReLU":
            self.activation = self._activation_layer_factory.get_layer(activation)
        else:
            self.activation = self._activation_layer_factory.get_layer(activation, True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet3D(torch.nn.Module):
    """
    3D ResNet adaptation from the original ResNet paper `"Deep Residual Learning for Image Recognition"
        <https://arxiv.org/abs/1512.03385>`.
    """

    def __init__(self, block: torch.nn.Module, n_blocks_per_layer: list, config: ModelConfiguration):
        """
        ResNet 3D model initializer.
        Args:
            block (:obj:`torch.nn.Module`): The desired block type (Basic or Bottleneck).
            n_blocks_per_layer (list):
            config:
        """
        super(ResNet3D, self).__init__()
        self._activation_layer_factory = ActivationFunctionsFactory()

        if config.num_groups is not None:
            norm_layer = torch.nn.GroupNorm
        else:
            norm_layer = torch.nn.BatchNorm3d

        self.norm_layer = norm_layer
        self.num_groups = config.num_groups
        self.activation_fn = config.activation
        self.groups = config.conv_groups
        self.base_width = config.width_per_group
        self.replace_stride_with_dilation = config.replace_stride_with_dilation
        self.inplanes = 64
        self.dilation = 1

        if self.replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        else:
            replace_stride_with_dilation = config.replace_stride_with_dilation

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(config.replace_stride_with_dilation))

        self.conv1 = torch.nn.Conv3d(config.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                     bias=False)

        if config.num_groups is not None:
            self.norm1 = norm_layer(self.num_groups, self.inplanes)
        else:
            self.norm1 = norm_layer(self.inplanes)

        if self.activation_fn == "PReLU":
            self.activation = self._activation_layer_factory.get_layer(self.activation_fn)
        else:
            self.activation = self._activation_layer_factory.get_layer(self.activation_fn, True)

        self.maxpool = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, n_blocks_per_layer[0])
        self.layer2 = self._make_layer(block, 128, n_blocks_per_layer[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, n_blocks_per_layer[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, n_blocks_per_layer[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, config.out_channels)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm3d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if config.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.norm3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.norm.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        num_groups = self.num_groups
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if num_groups is not None:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = torch.nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(num_groups, planes * block.expansion),
                )
        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = torch.nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.activation_fn, num_groups))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, activation=self.activation_fn))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(block, layers, config):
    """
    Construct the ResNet 3D network.

    Args:
        block (:obj:`torch.nn.Module`): The kind of ResNet block to build the ResNet with.
        layers (list): A list of number of blocks per layer.
        config (:obj:`samitorch.config.model.ModelConfiguration`): The network configuration.

    Returns:
        torch.nn.Module: The ResNet 3D Network.
    """
    model = ResNet3D(block, layers, config)

    return model


def resnet18(config):
    """Constructs a ResNet-18 model.

    Args:
        config (:obj:`samitorch.config.model.ModelConfiguration`): The network configuration.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], config)


def resnet34(config):
    """Constructs a ResNet-34 model.

    Args:
        config (:obj:`samitorch.config.model.ModelConfiguration`): The network configuration.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], config)


def resnet50(config):
    """Constructs a ResNet-50 model.

    Args:
        config (:obj:`samitorch.config.model.ModelConfiguration`): The network configuration.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], config)


def resnet101(config):
    """Constructs a ResNet-101 model.

    Args:
        config (:obj:`samitorch.config.model.ModelConfiguration`): The network configuration.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], config)


def resnet152(config):
    """Constructs a ResNet-152 model.

    Args:
        config (:obj:`samitorch.config.model.ModelConfiguration`): The network configuration.
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], config)
