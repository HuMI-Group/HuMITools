import torch
from torch import cat
from torch import nn
from torch.nn import BatchNorm3d, Dropout3d
from torch.nn import ELU


class Alex_Downsampling_Block(nn.Module):
    # This is how we downsample between functions, for now it is average pooling
    def __init__(self, in_channels, channels_previous_layers, internal_channels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.pooling_level = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 1), padding=0, ceil_mode=True),
            ELU()
        )
        self.DownsamplingSQE_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=in_channels, out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=internal_channels, out_channels=internal_channels,
                      padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels)
        )
        self.DownsamplingSQE_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=(in_channels + channels_previous_layers),
                      out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
        )

    def forward(self, current_layer):
        x1 = self.pooling_level(current_layer)
        x2 = self.DownsamplingSQE_part_1(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.DownsamplingSQE_part_2(x1)
        return x1


class Alex_Intermediate_Block(nn.Module):
    def __init__(self, in_channels, channels_previous_layers, internal_channels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.Intermediate_SQE_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=in_channels, out_channels=internal_channels, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=internal_channels, out_channels=internal_channels,
                      padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=internal_channels, out_channels=internal_channels,
                      padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels)
        )
        self.Intermediate_SQE_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=(in_channels + channels_previous_layers),
                      out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels)
        )

    def forward(self, current_layer):
        x1 = self.Intermediate_SQE_part_1(current_layer)
        x1 = torch.cat((x1, current_layer), dim=1)
        x1 = self.Intermediate_SQE_part_2(x1)
        return x1


class Alex_Upsampling_Block(nn.Module):
    def __init__(self, in_channels, channels_previous_layers, channels_downsampling_output, internal_channels,
                 dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.Upsampling_Block = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 1)), ELU())
        self.UpsamplingSQE_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=in_channels, out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=internal_channels, out_channels=internal_channels,
                      padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels)
        )
        self.UpsamplingSQE_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,),
                      in_channels=(in_channels + channels_previous_layers + channels_downsampling_output),
                      out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
        )

    def forward(self, current_layer, downsampling_layer):
        x1 = self.Upsampling_Block(current_layer)
        x2 = self.UpsamplingSQE_part_1(x1)
        x1 = cat((x1, x2, downsampling_layer), dim=1)
        x1 = self.UpsamplingSQE_part_2(x1)
        return x1


class Alex_Downsampling_Block_v2(nn.Module):
    # This is how we downsample between functions, for now it is average pooling
    def __init__(self, in_channels, channels_previous_layers, internal_channels, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.pooling_level = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0, ceil_mode=False,
                                          return_indices=True)

        self.DownsamplingSQE_part_1 = nn.Sequential(
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=in_channels, out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=internal_channels, out_channels=internal_channels,
                      padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels)
        )
        self.DownsamplingSQE_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=(in_channels + channels_previous_layers),
                      out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
        )

    def forward(self, current_layer):
        x1, indices = self.pooling_level(current_layer)
        x2 = self.DownsamplingSQE_part_1(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.DownsamplingSQE_part_2(x1)
        return x1, indices


class Alex_Upsampling_Block_v2(nn.Module):
    def __init__(self, in_channels, channels_previous_layers, channels_downsampling_output, internal_channels,
                 dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.Upsampling_Block = nn.MaxUnpool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.UpsamplingSQE_part_1 = nn.Sequential(
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=in_channels, out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=internal_channels, out_channels=internal_channels,
                      padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels)
        )
        self.UpsamplingSQE_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,),
                      in_channels=(in_channels + channels_previous_layers + channels_downsampling_output),
                      out_channels=internal_channels, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=internal_channels),
        )

    def forward(self, current_layer, downsampling_layer, indices):
        x1 = self.Upsampling_Block(current_layer, indices)
        x2 = self.UpsamplingSQE_part_1(x1)
        x1 = cat((x1, x2, downsampling_layer), dim=1)
        x1 = self.UpsamplingSQE_part_2(x1)
        return x1


class batchnorm_and_relu(nn.Module):
    def __init__(self, inchannel):
        super().__init__()

        self.bn = nn.BatchNorm3d(num_features=inchannel, affine=False)
        self.relu = nn.modules.activation.ReLU()

    def forward(self, input):
        x = self.bn(input)
        x = self.relu(x)
        return x


# do batchnorm and relu activation for number of in features
class downsampling(nn.Module):
    def __init__(self):
        super(downsampling, self).__init__()
        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)

    def forward(self, input):
        output = self.down(input)
        return output


class upsampling(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(upsampling, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels=inchannel, out_channels=outchannel, kernel_size=2, stride=(2, 2, 1),
                                     padding=0)

    def forward(self, input):
        output = self.up(input)
        return output


class unetblock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(unetblock, self).__init__()

        self.middleout = int(((
                                      outchannel - inchannel) / 2) + inchannel)  # output of the first should be half of the fifference between in and out of whole block

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=self.middleout, kernel_size=3, padding='same'),
            batchnorm_and_relu(self.middleout))
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=self.middleout, out_channels=outchannel, kernel_size=3, padding='same'),
            batchnorm_and_relu(outchannel))

    def forward(self, input):
        output1 = self.block1(input)
        output = self.block2(output1)
        return output


class resblock(nn.Module):
    def __init__(self, inchannel, outchannel, firstblock):
        super(resblock, self).__init__()
        self.middle_out = int(((outchannel - inchannel) / 2) + inchannel)
        if firstblock:
            self.block1 = nn.Sequential(nn.Conv3d(in_channels=inchannel, out_channels=self.middle_out, kernel_size=3,
                                                  padding='same'))  # first block has no batchnorm and relu at start
        else:
            self.block1 = nn.Sequential(batchnorm_and_relu(inchannel),
                                        nn.Conv3d(in_channels=inchannel, out_channels=self.middle_out, kernel_size=3,
                                                  padding='same'))

        self.block2 = nn.Sequential(batchnorm_and_relu(self.middle_out),
                                    nn.Conv3d(in_channels=self.middle_out, out_channels=outchannel, kernel_size=3,
                                              padding='same'))
        # residual (identity mapping)
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, padding=0),
            nn.BatchNorm3d(num_features=outchannel, affine=False))

    def forward(self, input):
        output1 = self.block1(input)
        output2 = self.block2(output1)
        r = self.residual(input)
        skip = torch.add(output2, r)
        return skip


#

class multiresblock(nn.Module):
    def __init__(self, inchannel, outchannelofmultiresblock, firstblock):  # , alpha=1):
        super(multiresblock, self).__init__()
        if inchannel < outchannelofmultiresblock:  # encoder
            self.out1 = int(outchannelofmultiresblock / 6)
            self.out2 = int(outchannelofmultiresblock / 3)
            self.out3 = outchannelofmultiresblock - self.out1 - self.out2
        else:  # decoder
            self.out3 = int(outchannelofmultiresblock / 6)
            self.out2 = int(outchannelofmultiresblock / 3)
            self.out1 = outchannelofmultiresblock - self.out2 - self.out3

        if firstblock:
            self.bn_relu1 = None
        else:
            self.bn_relu1 = batchnorm_and_relu(inchannel)

        self.con3x3 = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=self.out1, kernel_size=3, padding='same'))
        # self.con5x5 = nn.Sequential(nn.Conv3d(in_channels=int(self.W/6), out_channels=int(self.W/3), kernel_size=3, padding='same'),  batchnorm_and_relu(int(self.W/3)))#increase filters by 2

        self.con5x5 = nn.Sequential(batchnorm_and_relu(self.out1),
                                    nn.Conv3d(in_channels=self.out1, out_channels=self.out2, kernel_size=3,
                                              padding='same'))
        self.con7x7 = nn.Sequential(batchnorm_and_relu(self.out2),
                                    nn.Conv3d(in_channels=self.out2, out_channels=self.out3, kernel_size=3,
                                              padding='same'))

        # residual (identity mapping)
        # self.residual = nn.Sequential(nn.Conv3d(in_channels=inchannel, out_channels=int(self.W/6) + int(self.W/3) + int(self.W/2), kernel_size=1, padding=0),  nn.BatchNorm3d(num_features=int(int(self.W/6)) + int(self.W/3) + int(self.W/2), affine=False))
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=outchannelofmultiresblock, kernel_size=1, padding=0),
            nn.BatchNorm3d(num_features=outchannelofmultiresblock, affine=False))

    def forward(self, input):
        try:
            x = self.bn_relu1(input)
        except:
            x = input  # first block of network needs no bn and relu at start

        c3x3 = self.con3x3(x)
        c5x5 = self.con5x5(c3x3)
        c7x7 = self.con7x7(c5x5)
        concat = torch.cat((c3x3, c5x5, c7x7), 1)
        r = self.residual(input)
        skip = torch.add(concat, r)
        return skip


class respath(nn.Module):
    def __init__(self, inchannel, outchannel, depth):
        super(respath, self).__init__()
        # depth describes the number of convolutions
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.depth = depth

        self.con1 = nn.Sequential(batchnorm_and_relu(inchannel),
                                  nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=3,
                                            padding='same'))

        self.con = nn.Sequential(batchnorm_and_relu(outchannel),
                                 nn.Conv3d(in_channels=outchannel, out_channels=outchannel, kernel_size=3,
                                           padding='same'))

        self.residual1 = nn.Sequential(
            nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, padding=0),
            nn.BatchNorm3d(num_features=outchannel, affine=False))
        self.batchnorm = nn.BatchNorm3d(num_features=outchannel, affine=False)
        # self.residual = nn.Sequential(nn.Conv3d(in_channels=outchannel, out_channels=outchannel, kernel_size=1, padding=0),
        # nn.BatchNorm3d(num_features=outchannel, affine=False))

    def forward(self, input):
        if self.inchannel == self.outchannel:
            residual = self.batchnorm(input)  # if output is same as there is no need fpr a convolution on residual
            con = self.con(input)
            output = con + residual
        else:
            residual = self.residual1(input)
            con = self.con1(input)
            output = con + residual

        for i in range(self.depth - 1):
            residual = self.batchnorm(output)
            con = self.con(output)
            output = con + residual
        return output


class denslayer(nn.Module):
    def __init__(self, inchannel, outchannels):
        super(denslayer, self).__init__()
        # each encoder convolution layer expands channellayer by filters, each decoder decreases by -filters
        self.bn_relu = batchnorm_and_relu(inchannel)
        self.con = nn.Conv3d(in_channels=inchannel, out_channels=outchannels, kernel_size=3, padding='same')

    def forward(self, input):
        bnrelu = self.bn_relu(input)
        conv = self.con(bnrelu)
        return conv


class densblock(nn.Module):
    def __init__(self, filters, repetitions, inchannel):
        super(densblock, self).__init__()
        self.filters = filters  # numper to increase per layer
        self.repetitions = repetitions  # number of layers per block
        self.layerlist = self.createlayerlist(filters, repetitions,
                                              inchannel)  # list with all layers and needed in and outchannel

    def createlayerlist(self, filters, repetitions, inchannel):  # function for init
        layer_list = []
        currentinchannel = inchannel
        for i in range(repetitions):
            currentoutchannel = currentinchannel + filters  # the output is expanded by filter (mostly 16 or 12)
            layer_list.append(denslayer(currentinchannel, currentoutchannel))
            currentinchannel = currentoutchannel + currentinchannel  # for the next layer, the inchannel is the previous outchannelsize plus previous inchqannel, because they will be concatenated
        return layer_list

    def forward(self, input):
        layerinput = input
        repetitions = self.repetitions
        for i in range(repetitions):  # densblock with 3 convolutions
            layer = self.layerlist[i]
            layeroutput = layer(layerinput)
            layerinput = torch.cat((layerinput, layeroutput), 1)  # concat previous layers inputs and outputs
            # output of densblock should not concat input
            if i == 0:
                output = layeroutput
            else:
                output = torch.cat((output, layeroutput), 1)
        return output


class transitiondown(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(transitiondown, self).__init__()
        self.bn_relu = batchnorm_and_relu(inchannel)
        self.con = nn.Conv3d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm3d(num_features=outchannel, affine=False)
        # self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=(0))

    def forward(self, input):
        bnrelu = self.bn_relu(input)
        conv = self.con(bnrelu)
        conv = self.bn(conv)
        # norm = self.bn(conv)
        # output = self.down(conv)
        return conv  # ,output
