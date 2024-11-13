import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class CrossAttentionModel(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        """Model with cross-attention between two datasets."""
        super(CrossAttentionModel, self).__init__()
        # ResNet models for both datasets
        # wavelet
        self.resnet1 = ResNet50(2, 3)
        # images
        self.resnet2 = models.resnet50(pretrained=False)
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])  # remove F
        # self.resnet2.conv1 = nn.Conv2d(
        #     12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        # )

        # Cross-attention layer
        self.cross_attention1 = CrossAttention(feature_dim, num_heads=num_heads)
        self.cross_attention2 = CrossAttention(feature_dim, num_heads=num_heads)

        # Classifier after attention
        self.fc = nn.Linear(
            feature_dim + feature_dim, 2
        )  # Assume binary classification

    def forward(self, image1, image2):
        image2 = image2.permute([0, 3, 1, 2])

        # Extract features from both images
        features1 = self.resnet1(image1)  # Output shape: (batch_size, feature_dim)
        features2 = self.resnet2(image2)  # Output shape: (batch_size, feature_dim)

        # Flatten the features
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        # Reshape for cross-attention (add sequence dimension for attention to work)
        features1 = features1.unsqueeze(0)  # Shape: (1, batch_size, feature_dim)
        features2 = features2.unsqueeze(0)  # Shape: (1, batch_size, feature_dim)

        # Apply cross-attention: image1 features attend to image2 features, and vice versa
        attn_output1 = self.cross_attention1(features1, features2, features2)
        attn_output2 = self.cross_attention2(features2, features1, features1)

        # Remove sequence dimension and combine both attention outputs
        combined_features = torch.cat(
            [attn_output1.squeeze(0), attn_output2.squeeze(0)], dim=1
        )

        # Pass through a fully connected layer for classification
        output = self.fc(combined_features)

        return output


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        """Initialize cross-attention mechanism.

        Args:
            embed_dim (int): Dimensionality of the embeddings.
            num_heads (int): Number of attention heads.
        """
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        """Forward pass for cross-attention.

        Args:
            query (Tensor): The query tensor, typically from dataset1's features.
            key (Tensor): The key tensor, typically from dataset2's features.
            value (Tensor): The value tensor, typically from dataset2's features.

        Returns:
            Tensor: The output of the cross-attention mechanism.
        """
        # Apply multi-head attention: Query attends to the key-value pairs.
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output


class LateFusionResNet(nn.Module):
    def __init__(self, num_classes):
        super(LateFusionResNet, self).__init__()

        # Load two ResNet models (e.g., ResNet18 and ResNet50)
        # wavelet
        self.resnet1 = ResNet50(2, 3)
        # images
        self.resnet2 = models.resnet50(pretrained=False)

        # Remove the final fully connected layers from both
        # self.resnet1 = nn.Sequential(
        # *list(self.resnet1.children())[:-1]
        # )  # remove FC layer
        self.resnet2 = nn.Sequential(
            *list(self.resnet2.children())[:-1]
        )  # remove FC layer

        # Fusion layer: Concatenate features from both models
        self.fc = nn.Linear(
            2048 + 2048, num_classes
        )  # 512 for ResNet18 and 2048 for ResNet50

    def forward(self, x1, x2):
        # Pass inputs through both ResNet models
        features1 = self.resnet1(x1)  # Output of ResNet50
        # print(x2.shape)
        x2 = x2.permute([0, 3, 1, 2])
        features2 = self.resnet2(x2)  # Output of ResNet50

        # Flatten the features from both ResNets
        features1 = features1.view(features1.size(0), -1)
        features2 = features2.view(features2.size(0), -1)

        # Concatenate the features (late fusion)
        combined_features = torch.cat((features1, features2), dim=1)

        # Final classification layer
        output = self.fc(combined_features)
        return output


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(
        self, ResBlock, layer_list, num_classes, num_channels=1, feature="packets"
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.feature = feature
        if self.feature == "packets":
            # Modify input layer for packets
            self.conv1 = nn.Conv2d(
                4 * num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        elif self.feature == "image":
            # No modification needed for image input
            self.conv1 = nn.Conv2d(
                num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            raise ValueError(
                "Invalid feature type. Supported types are 'packets' and 'image'."
            )

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        shape = x.shape
        # print(shape)
        # Batch size, height, width, packets, channels
        x = x.permute([0, 2, 3, 1, 4])  # we don't ned to do that with combined dataset
        # Reshape to [batch_size, height, width, packets*channels]
        to_net = x.reshape([shape[0], shape[2], shape[3], shape[1] * shape[4]])
        to_net = to_net.permute([0, 3, 1, 2])  # change back to to_net for normal net
        # Apply initial convolutional layer
        x = self.relu(self.batch_norm1(self.conv1(to_net)))
        # x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.max_pool(x)

        # Proceed with the rest of the ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Perform average pooling and flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # remove for late fusion
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=192):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)


# Now modify the ResNet class to include the cross-attention mechanism
