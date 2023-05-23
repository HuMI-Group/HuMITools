import torch
import torch.nn.functional as F
from torch import cat
from torch import nn
from torch.nn import BatchNorm3d, Dropout3d
from torch.nn import ELU, Softmax

from humi_pytorch.modules import Alex_Upsampling_Block, Alex_Downsampling_Block, Alex_Intermediate_Block, \
    Alex_Upsampling_Block_v2, Alex_Downsampling_Block_v2, batchnorm_and_relu, upsampling, unetblock, resblock, \
    multiresblock, respath, densblock, transitiondown


class basic_network(nn.Module):

    def __init__(self, numberlabels):
        super(basic_network, self).__init__()
        """Initialize neural net layers."""

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3, padding='same'),
            nn.modules.activation.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=12, out_channels=36, kernel_size=3, padding='same'),
            nn.modules.activation.ReLU())
        self.conv3 = nn.Sequential(
            # nn.Conv3d(in_channels=36, out_channels=8, kernel_size=3, padding='same'),
            nn.Conv3d(in_channels=36, out_channels=numberlabels + 1, kernel_size=3, padding='same'),
            nn.modules.activation.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)
        output = F.softmax(x, dim=1)

        return output


class unet(nn.Module):
    # general architecture taken from: arXiv:1606.06650
    # Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger
    def __init__(self, numberlabels):
        super(unet, self).__init__()
        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)

        self.block1 = unetblock(inchannel=1, outchannel=64)

        self.block2 = unetblock(inchannel=64, outchannel=128)
        self.Block3 = unetblock(inchannel=128, outchannel=256)
        self.Block4 = unetblock(inchannel=256, outchannel=256)

        self.TransposeBlock1 = upsampling(inchannel=256, outchannel=256)
        self.Block5 = unetblock(inchannel=512, outchannel=128)
        self.TransposeBlock2 = upsampling(inchannel=128, outchannel=128)

        self.Block6 = unetblock(inchannel=256, outchannel=64)
        self.TransposeBlock3 = upsampling(inchannel=64, outchannel=64)
        self.Block7 = unetblock(inchannel=128, outchannel=64)

        self.last_layer = nn.Conv3d(in_channels=64, out_channels=numberlabels + 1, kernel_size=3, padding='same')

    def forward(self, x):
        x1 = self.block1(x)  # 4,64,144,144,22
        maxpooling1 = self.down(x1)  # 4,64,72,72,21

        x2 = self.block2(maxpooling1)  # 4,128,72,72,21
        maxpooling2 = self.down(x2)  # 4,128,36,36,20

        x3 = self.Block3(maxpooling2)  # 4,256,36,36,20
        maxpooling3 = self.down(x3)  # 4,256,18,18,19

        # bridge
        x4 = self.Block4(maxpooling3)  # 4,256,18,18,19
        transposeb4 = self.TransposeBlock1(x4)  # 4,256,36,36,20

        # decoder
        inputb5 = torch.cat((x3, transposeb4), 1)  # 4,512,36,36,20
        x5 = self.Block5(inputb5)  # 4,128,36,36,20
        transposeb5 = self.TransposeBlock2(x5)  # 4,128,72,72,21

        inputb6 = torch.cat((x2, transposeb5), 1)  # 4,256,72,72,21
        x6 = self.Block6(inputb6)  # 4,64,72,72,21
        transpose6 = self.TransposeBlock3(x6)  # 4,64,144,144,22

        inputb7 = torch.cat((x1, transpose6), 1)  # 4,128,144,144,22
        b7 = self.Block7(inputb7)  # 4,64,144,144,22

        x = self.last_layer(b7)  # 4,9,144,144,22

        # output = x # F.softmax(x, dim = 0)
        output = F.softmax(x, dim=1)
        return output


class multiunet(nn.Module):
    # general architecture taken from: arXiv:1606.06650
    # uses two inputs instead of the conventional one for possible
    def __init__(self):
        super(multiunet, self).__init__()
        """Initialize neural net layers."""
        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)

        self.block1 = unetblock(inchannel=1, outchannel=64)

        self.block2 = unetblock(inchannel=64, outchannel=128)
        self.Block3 = unetblock(inchannel=128, outchannel=256)
        self.Block4 = unetblock(inchannel=512, outchannel=512)

        self.TransposeBlock1 = upsampling(inchannel=512, outchannel=512)
        self.Block5 = unetblock(inchannel=1024, outchannel=128)

        self.TransposeBlock2 = upsampling(inchannel=128, outchannel=128)

        self.Block6 = unetblock(inchannel=384, outchannel=64)
        self.TransposeBlock3 = upsampling(inchannel=64, outchannel=64)

        self.Block7 = unetblock(inchannel=192, outchannel=64)

        self.last_layer = nn.Conv3d(in_channels=64, out_channels=9, kernel_size=3, padding='same')

    def forward(self, input1, input2):
        skip1_1 = self.block1(input1)  # 4,64,144,144,22
        maxpooling1_1 = self.down(skip1_1)  # 4,64,72,72,21

        skip1_2 = self.block1(input2)  # 4,64,144,144,22
        maxpooling1_2 = self.down(skip1_2)  # 4,64,72,72,21

        skip2_1 = self.block2(maxpooling1_1)  # 4,128,72,72,21
        maxpooling2_1 = self.down(skip2_1)  # 4,128,36,36,20

        skip2_2 = self.block2(maxpooling1_2)  # 4,128,72,72,21
        maxpooling2_2 = self.down(skip2_2)  # 4,128,36,36,20

        skip3_1 = self.Block3(maxpooling2_1)  # 4,256,36,36,20
        maxpooling3_1 = self.down(skip3_1)  # 4,256,18,18,19

        skip3_2 = self.Block3(maxpooling2_2)  # 4,256,36,36,20
        maxpooling3_2 = self.down(skip3_2)  # 4,256,18,18,19

        input_bridge = torch.cat((maxpooling3_1, maxpooling3_2), 1)  # 4,512,18,18,19

        x4 = self.Block4(input_bridge)  # 4,512,18,18,19

        transpose = self.TransposeBlock1(x4)  # 4,512,36,36,20
        x = torch.cat((skip3_1, skip3_2, transpose), 1)  # 4,1024,36,36,20

        x = self.Block5(x)  # 4,128,36,36,20
        transpose = self.TransposeBlock2(x)  # 4,128,72,72,21
        x = torch.cat((skip2_1, skip2_2, transpose), 1)  # 4,384,36,36,20
        x = self.Block6(x)  # 4,64,72,72,21
        transpose = self.TransposeBlock3(x)  # 4,64,144,144,22
        x = torch.cat((skip1_1, skip1_2, transpose), 1)  # 4,192,144,144,22
        x = self.Block7(x)  # 4,64,144,144,22
        x = self.last_layer(x)  # 4,9,144,144,22

        # output = x # F.softmax(x, dim = 0)
        output = F.softmax(x, dim=1)  # 4,9,144,144,22
        return output


######################################
############resunet###################
######################################

class resunet(nn.Module):
    # general architecture as in: arXiv:1904.00592
    # Foivos I. Diakogiannis, François Waldner, Peter Caccetta, Chen Wu
    # with adaptations to decrease memory usage in 3d
    def __init__(self, numberlabels):
        super().__init__()
        """Initialize neural net layers."""
        # block1
        # self.resblock1 = resblock(1, 64,groups=1)
        # first block needs no batchnorm and relu at start
        self.resblock1 = resblock(inchannel=1, outchannel=64, firstblock=True)

        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)
        # block2
        self.resblock2 = resblock(64, 128, False)
        # block3
        self.resblock3 = resblock(128, 256, False)

        # block4
        self.resblock4 = resblock(256, 256, False)
        self.upsample_b4 = upsampling(256, 256)

        # block 5
        self.resblock5 = resblock(512, 128, False)
        self.upsample_b5 = upsampling(128, 128)

        # block6
        self.resblock6 = resblock(256, 64, False)
        self.upsample_b6 = upsampling(64, 64)

        # block7
        self.resblock7 = resblock(128, 64, False)

        self.batchandactivation = batchnorm_and_relu(64)
        self.bn = nn.BatchNorm3d(num_features=64, affine=False)
        # last layer
        self.last_layer = nn.Sequential(
            batchnorm_and_relu(64),
            nn.Conv3d(in_channels=64, out_channels=numberlabels + 1, kernel_size=3, padding='same'))

    ############resunet###################
    def forward(self, x):
        # encoding
        # block1
        skip1 = self.resblock1(x)  # 4,64,144,144,22
        output1 = self.down(skip1)  # 4,64,72,72,21

        # first block needs no batchnorm and relu at start
        # block2
        skip2 = self.resblock2(output1)  # 4,128,72,72,21
        output2 = self.down(skip2)  # 4,128,36,36,20

        # block3
        skip3 = self.resblock3(output2)  # 4,256,36,36,20
        output3 = self.down(skip3)  # 4,256,18,18,19

        # block4 (bottom)
        skip4 = self.resblock4(output3)  # 4,256,18,18,19
        output4 = self.upsample_b4(skip4)  # 4,256,36,36,20

        # decoding block5
        inputb5 = torch.cat((output4, skip3), 1)  # 4,512,36,36,20
        b5 = self.resblock5(inputb5)  # 4,128,36,36,20
        output5 = self.upsample_b5(b5)  # 4,128,72,72,21

        # decoding block6
        inputb6 = torch.cat((output5, skip2), 1)  # 4,256,72,72,21
        b6 = self.resblock6(inputb6)  # 4,64,72,72,21
        output6 = self.upsample_b6(b6)  # 4,64,144,144,22

        # decoding block7
        inputb7 = torch.cat((output6, skip1), 1)  # 4,128,144,144,22
        b7 = self.resblock7(inputb7)  # 4,64,144,144,22

        x = self.last_layer(b7)  # 4,64,144,144,22
        output = F.softmax(x, dim=1)  # 4,9,144,144,22
        return output


######################################
#######multiresunet###################
##resunet whith multiple input IMAGES#
######################################

class multiresunet(nn.Module):
    # general architecture as in: arXiv:1904.00592
    # Foivos I. Diakogiannis, François Waldner, Peter Caccetta, Chen Wu
    # with adaptations to decrease memory usage in 3d
    # takes two inputs and has seperate decoder paths

    def __init__(self, numberlabels):
        super().__init__()
        """Initialize neural net layers."""
        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)

        # first block needs no batchnorm and relu at start
        # first block
        self.firstblock = resblock(1, 64, True)

        self.identitymapping = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=1, padding=0)

        # block2
        self.resblock2 = resblock(64, 128, False)
        # block3
        self.resblock3 = resblock(128, 256, False)

        # block4 (bridge)
        self.resblock4 = resblock(512, 512, False)  # contains concatinated encoderstreams
        self.upsample_b4 = upsampling(512, 512)

        # decoder
        # block 5
        self.resblock5 = resblock(1024, 128, False)
        self.upsample_b5 = upsampling(128, 128)

        # block6
        self.resblock6 = resblock(384, 64, False)
        self.upsample_b6 = upsampling(64, 64)

        # block7
        self.resblock7 = resblock(192, 64, False)

        # last layer
        self.last_layer = nn.Sequential(
            batchnorm_and_relu(64),
            nn.Conv3d(in_channels=64, out_channels=numberlabels + 1, kernel_size=3, padding='same'))

    #######multiresunet###################

    def forward(self, input1, input2):
        ###############encoder############################
        # resblock 1.1
        skip1_1 = self.firstblock(input1)  # 4,64,144,144,22
        # maxpooling 1.1
        output1_1 = self.down(skip1_1)  # 4,64,72,72,21

        # resblock 1.2
        skip1_2 = self.firstblock(input2)  # 4,64,144,144,22
        # maxpooling 1.2
        output1_2 = self.down(skip1_2)  # 4,64,72,72,21

        # resblock 2.1
        skip2_1 = self.resblock2(output1_1)  # 4,128,72,72,21
        # maxpooling 2.1
        output2_1 = self.down(skip2_1)  # 4,128,36,36,20

        # resblock 2.2
        skip2_2 = self.resblock2(output1_2)  # 4,128,72,72,21
        # maxpooling 2.2
        output2_2 = self.down(skip2_2)  # 4,128,36,36,20

        # resblock 3.1
        skip3_1 = self.resblock3(output2_1)  # 4,256,36,36,20
        # maxpooling 3.1
        output3_1 = self.down(skip3_1)  # 4,256,18,18,19

        # resblock 3.2
        skip3_2 = self.resblock3(output2_2)  # 4,256,36,36,20
        # maxpooling 3.2
        output3_2 = self.down(skip3_2)  # 4,256,18,18,19

        # encoder is finished

        ################bridge############################
        concatinatedencoders = torch.cat((output3_1, output3_2), 1)  # 4,512,18,18,19
        resblock4 = self.resblock4(concatinatedencoders)  # 4,512,18,18,19
        # upsample bridge
        output = self.upsample_b4(resblock4)  # 4,512,36,36,20

        ###############decoder############################
        # decoding block5

        inputb5 = torch.cat((output, skip3_1, skip3_2), 1)  # 4,1024,36,36,20
        b5 = self.resblock5(inputb5)  # 4,128,36,36,20
        output5 = self.upsample_b5(b5)  # 4,128,72,72,21

        # decoding block6
        inputb6 = torch.cat((output5, skip2_1, skip2_2), 1)  # 4,384,72,72,21
        b6 = self.resblock6(inputb6)  # 4,64,72,72,21
        output6 = self.upsample_b6(b6)  # 4,64,144,144,22

        # decoding block7
        inputb7 = torch.cat((output6, skip1_1, skip1_2), 1)  # 4,192,144,144,22
        b7 = self.resblock7(inputb7)  # 4,64,144,144,22

        x = self.last_layer(b7)  # 4,9,144,144,22
        output = F.softmax(x, dim=1)
        return output


######################################
###########denseunet###################
######################################


class denseunet(nn.Module):
    # general architecture taken from: doi:10.1109/JBHI.2019.2912935
    # with adaptations for reduced memory usage to counterbalance the increased memory requirements

    def __init__(self, numberlabels):
        super().__init__()
        """Initialize neural net layers."""
        self.firstlayerdensblock1 = nn.Conv3d(in_channels=1, out_channels=13, kernel_size=3, padding='same')

        self.secondlayerdensblock1 = nn.Sequential(
            batchnorm_and_relu(14),
            nn.Conv3d(in_channels=14, out_channels=26, kernel_size=3, padding='same')
        )
        self.comp1 = transitiondown(40, 32)
        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)

        # block encoder
        self.densblock2 = densblock(filters=12, repetitions=2, inchannel=32)
        self.comp2 = transitiondown(164, 64)

        self.densblock3 = densblock(filters=12, repetitions=2, inchannel=64)
        self.comp3 = transitiondown(292, 128)

        self.densblock4 = densblock(filters=12, repetitions=2, inchannel=128)
        self.transitionup_b4 = upsampling(420, 128)

        # blocks decoder
        self.densblock5 = densblock(filters=12, repetitions=2, inchannel=256)
        self.transitionup_b5 = upsampling(804, 64)

        self.densblock6 = densblock(filters=12, repetitions=2, inchannel=128)
        self.transitionup_b6 = upsampling(420, 32)

        self.densblock7 = densblock(filters=12, repetitions=1, inchannel=64)

        # last layer
        self.last_layers = nn.Sequential(
            batchnorm_and_relu(76),
            nn.Conv3d(in_channels=76, out_channels=32, kernel_size=3, padding='same'),
            batchnorm_and_relu(32),
            nn.Conv3d(in_channels=32, out_channels=numberlabels + 1, kernel_size=3, padding='same')
        )

    ######mdenseunet###################
    def forward(self, input):
        # encoding
        # densblock1
        first = self.firstlayerdensblock1(input)  # 4,13,144,144,22
        concatinputandfirst = torch.cat((first, input), 1)  # 4,14,144,144,22
        second = self.secondlayerdensblock1(concatinputandfirst)  # 4,26,144,144,22
        encoderconcat1_1 = torch.cat((second, first, input), 1)  # 4,40,144,144,22
        skip1 = self.comp1(encoderconcat1_1)  # 4,32,144,144,22
        # maxpooling 1
        output1 = self.down(skip1)  # 4,32,72,72,21

        # densblock 2
        db2 = self.densblock2(output1)  # densblock out: 4,132,72,72,21
        encoderconcat2 = torch.cat((db2, output1), 1)  # 4,164,72,72,21 #concatinates input and output of densblock
        skip2 = self.comp2(encoderconcat2)  # 4,64,72,72,21
        # maxpooling 2.1
        output2 = self.down(skip2)  # 4,64,36,36,20 maxpooling

        # densblock 3.1
        db3 = self.densblock3(output2)  # densblock out: #4,228,36,36,20
        encoderconcat3 = torch.cat((db3, output2), 1)  # 4,292,36,36,20 #concatinates input and output of densblock
        skip3 = self.comp3(encoderconcat3)  # 4,128,36,36,20
        # maxpooling 3.1
        output3 = self.down(skip3)  # 4,128,18,18,19 maxpooling

        ##########bridge (b4)#########
        bridge_db4 = self.densblock4(output3)  # 4,420,18,18,19
        # upconvolution bridge (b4)
        out_bridge_db4 = self.transitionup_b4(bridge_db4)  # 4,128,36,36,20
        ###########################

        # concat bridge and encoder skips
        inputb5 = torch.cat((out_bridge_db4, skip3), 1)  # 4,256,36,36,20
        # densblock 5
        b5 = self.densblock5(inputb5)  # 4,804,36,36,20
        # upconvolution b5
        output5 = self.transitionup_b5(b5)  # 4,64,72,72,21

        inputb6 = torch.cat((output5, skip2), 1)  # 4,128,72,72,21
        # densblock 6
        b6 = self.densblock6(inputb6)  # 4,420,72,72,21
        # upconvolution b6
        output6 = self.transitionup_b6(b6)  # 4,32,144,144,22

        inputb7 = torch.cat((output6, skip1), 1)  # 4,64,144,144,22
        # densblock 7

        b7 = self.densblock7(inputb7)  # 4,76,144,144,22

        x = self.last_layers(b7)  ##4,9,144,144,22
        output = F.softmax(x, dim=1)
        return output


######################################
######multidenseunet###################
######################################

class multidenseunet(nn.Module):
    # general architecture taken from: doi:10.1109/JBHI.2019.2912935
    # with adaptations for reduced memory usage to counterbalance the increased memory requirements
    # with adapatations to allow for multiple input signals (e.g. a T1 weighted and a Fat_Fraction weighted image)
    def __init__(self):
        super().__init__()
        """Initialize neural net layers."""
        self.firstlayerdensblock1 = nn.Conv3d(in_channels=1, out_channels=13, kernel_size=3, padding='same')

        self.secondlayerdensblock1 = nn.Sequential(
            batchnorm_and_relu(14),
            nn.Conv3d(in_channels=14, out_channels=26, kernel_size=3, padding='same')
        )
        self.comp1 = transitiondown(40, 32)
        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)

        # block encoder
        self.densblock2 = densblock(filters=12, repetitions=2, inchannel=32)
        self.comp2 = transitiondown(164, 64)

        self.densblock3 = densblock(filters=12, repetitions=2, inchannel=64)
        self.comp3 = transitiondown(292, 128)

        self.densblock4 = densblock(filters=12, repetitions=2, inchannel=256)
        self.transitionup_b4 = upsampling(804, 128)

        # blocks decoder
        self.densblock5 = densblock(filters=12, repetitions=2, inchannel=384)
        self.transitionup_b5 = upsampling(1188, 64)

        self.densblock6 = densblock(filters=12, repetitions=2, inchannel=192)
        self.transitionup_b6 = upsampling(612, 32)

        self.densblock7 = densblock(filters=12, repetitions=1, inchannel=96)

        # last layer
        self.last_layers = nn.Sequential(
            batchnorm_and_relu(108),
            nn.Conv3d(in_channels=108, out_channels=32, kernel_size=3, padding='same'),
            batchnorm_and_relu(32),
            nn.Conv3d(in_channels=32, out_channels=9, kernel_size=3, padding='same')
        )

    ######multidenseunet###################
    def forward(self, input1, input2):
        # encoding
        # densblock1.1
        first1 = self.firstlayerdensblock1(input1)  # 4,13,144,144,22
        concatinputandfirst1 = torch.cat((first1, input1), 1)  # 4,14,144,144,22
        second1 = self.secondlayerdensblock1(concatinputandfirst1)  # 4,26,144,144,22
        encoderconcat1_1 = torch.cat((second1, first1, input1), 1)  # 4,40,144,144,22
        skip1_1 = self.comp1(encoderconcat1_1)  # 4,32,144,144,22
        # maxpooling 1_1
        output1_1 = self.down(skip1_1)  # 4,32,72,72,21

        # densblock1.2
        first2 = self.firstlayerdensblock1(input2)  # 4,13,144,144,22
        concatinputandfirst2 = torch.cat((first2, input2), 1)  # 4,14,144,144,22
        second2 = self.secondlayerdensblock1(concatinputandfirst2)  # 4,26,144,144,22
        encoderconcat1_2 = torch.cat((second2, first1, input2), 1)  # 4,40,144,144,22
        skip1_2 = self.comp1(encoderconcat1_2)  # 4,32,144,144,22
        # maxpooling 2.1
        output1_2 = self.down(skip1_2)  # 4,32,72,72,21

        # densblock 2.1
        db2_1 = self.densblock2(output1_1)  # densblock out: 4,132,72,72,21
        encoderconcat2_1 = torch.cat((db2_1, output1_1),
                                     1)  # 4,164,72,72,21 #concatinates input and output of densblock
        skip2_1 = self.comp2(encoderconcat2_1)  # 4,64,144,144,22
        # maxpooling 2.1
        output2_1 = self.down(skip2_1)  # 4,64,36,36,20 maxpooling

        # densblock 2.2
        db2_2 = self.densblock2(output1_2)  # densblock out: 4,132,72,72,21
        encoderconcat2_2 = torch.cat((db2_2, output1_2),
                                     1)  # 4,164,72,72,21 #concatinates input and output of densblock
        skip2_2 = self.comp2(encoderconcat2_2)  # 4,64,144,144,22
        # maxpooling 2.1
        output2_2 = self.down(skip2_2)  # 4,64,36,36,20 maxpooling

        # densblock 3.1
        db3_1 = self.densblock3(output2_1)  # densblock out: #4,228,36,36,20
        encoderconcat3_1 = torch.cat((db3_1, output2_1),
                                     1)  # 4,292,36,36,20 #concatinates input and output of densblock
        skip3_1 = self.comp3(encoderconcat3_1)  # 4,128,36,36,20
        # maxpooling 3.1
        output3_1 = self.down(skip3_1)  # 4,128,18,18,19 maxpooling

        # densblock 3.2
        db3_2 = self.densblock3(output2_2)  # densblock out: #4,228,36,36,20
        encoderconcat3_2 = torch.cat((db3_2, output2_2),
                                     1)  # 4,292,36,36,20 #concatinates input and output of densblock
        skip3_2 = self.comp3(encoderconcat3_2)  # 4,128,36,36,20
        # maxpooling 3.2
        output3_2 = self.down(skip3_2)  # 4,128,18,18,19 maxpooling

        ##########bridge (b4)#########
        concatencoder = torch.cat((output3_1, output3_2), 1)  # 4,256,18,18,19
        bridge_db4 = self.densblock4(concatencoder)  # 4,804,18,18,19
        # upconvolution bridge (b4)
        out_bridge_db4 = self.transitionup_b4(bridge_db4)  # 4,128,36,36,20
        ###########################

        # concat bridge and encoder skips
        inputb5 = torch.cat((out_bridge_db4, skip3_1, skip3_2), 1)  # 4,384,36,36,20
        # densblock 5
        b5 = self.densblock5(inputb5)  # 4,1188,36,36,20
        # upconvolution b5
        output5 = self.transitionup_b5(b5)  # 4,64,72,72,21

        inputb6 = torch.cat((output5, skip2_1, skip2_2), 1)  # 4,193,72,72,21
        # densblock 6
        b6 = self.densblock6(inputb6)  # 4,612,72,72,21
        # upconvolution b6
        output6 = self.transitionup_b6(b6)  # 4,32,144,144,22

        inputb7 = torch.cat((output6, skip1_1, skip1_2), 1)  # 4,96,144,144,22
        # densblock 7

        b7 = self.densblock7(inputb7)  # 4,108,144,144,22

        x = self.last_layers(b7)  # 4,108,144,144,22 -> #4,32,144,144,22 -> #4,9,144,144,22
        output = F.softmax(x, dim=1)
        return output


#######################################################
###########resunetwithmultiresblocks##################(multiresolution blocks)
#######################################################
class resunetwithmultiresblocks(nn.Module):
    #	This is taken from: arXiv:1902.04049
    #   and adapted to fit into limited VRAM
    def __init__(self, numberlabels):
        super().__init__()

        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)
        # block encoder
        self.multiresblock1 = multiresblock(inchannel=1, outchannelofmultiresblock=64, firstblock=True)
        self.respath1 = respath(inchannel=64, outchannel=64, depth=3)

        self.multiresblock2 = multiresblock(inchannel=64, outchannelofmultiresblock=128, firstblock=False)
        self.respath2 = respath(inchannel=128, outchannel=128, depth=2)

        self.multiresblock3 = multiresblock(inchannel=128, outchannelofmultiresblock=256, firstblock=False).to('cuda:1')
        self.respath3 = respath(inchannel=256, outchannel=256, depth=1)

        # bridge
        self.multiresblock4 = multiresblock(inchannel=256, outchannelofmultiresblock=256, firstblock=False).to('cuda:1')
        self.upsample1 = upsampling(256, 256)

        # decoder
        self.multiresblock5 = multiresblock(inchannel=512, outchannelofmultiresblock=128, firstblock=False).to('cuda:1')
        self.upsample2 = upsampling(128, 128)

        self.multiresblock6 = multiresblock(inchannel=256, outchannelofmultiresblock=64, firstblock=False).to('cuda:0')
        self.upsample3 = upsampling(64, 64)

        self.multiresblock7 = multiresblock(inchannel=128, outchannelofmultiresblock=64, firstblock=False).to('cuda:0')
        self.upsample_b6 = upsampling(64, 64)

        # last layer
        # last layer
        self.last_layer = nn.Sequential(
            batchnorm_and_relu(64),
            nn.Conv3d(in_channels=64, out_channels=numberlabels + 1, kernel_size=3, padding='same'))

    ##########resunetwithmultiresblocks###################
    def forward(self, input):
        # encoding
        skip1 = self.multiresblock1(input)  # 4,64,144,144,22
        output1 = self.down(skip1)  # 4,64,72,72,21
        respathskip1 = self.respath1(skip1)  # 4,64,144,144,22

        skip2 = self.multiresblock2(output1)  # 4,128,72,72,21
        output2 = self.down(skip2)  # 4,128,36,36,20
        respathskip2 = self.respath2(skip2)  # 4,128,72,72,21

        skip3 = self.multiresblock3(output2)  # 4,256,36,36,20
        output3 = self.down(skip3)  # 4,256,18,18,19
        respathskip3 = self.respath3(skip3)  # 4,256,36,36,20

        ##########bridge (b4)#########
        bridge = self.multiresblock4(output3)  # 4,256,18,18,19
        # upconvolution bridge (b4)
        out_bridge = self.upsample1(bridge)  # 4,256,36,36,20
        ###########################

        # concat bridge and encoder skips
        inputb5 = torch.cat((out_bridge, respathskip3), 1)  # 4,512,36,36,20
        b5 = self.multiresblock5(inputb5)  # 4,128,36,36,20
        output5 = self.upsample2(b5)  # 4,128,72,72,21

        # decoding block6
        inputb6 = torch.cat((output5, respathskip2), 1)  # 4,256,72,72,21
        b6 = self.multiresblock6(inputb6)  # 4,64,72,72,21
        output6 = self.upsample3(b6)  # 4,64,144,144,22

        # decoding block7
        inputb7 = torch.cat((output6, respathskip1), 1)  # 4,128,144,144,22
        b7 = self.multiresblock7(inputb7)  # 4,64,144,144,22

        x = self.last_layer(b7)  # 4,9,144,144,22
        output = F.softmax(x, dim=1)
        return output


#######################################################
######multiresunetwithmultiresblocks##################
#######################################################
class multiresunetwithmultiresblocks(nn.Module):
    #	This is taken from: arXiv:1902.04049
    #   and adapted to fit into limited VRAM
    #   additional adaptations to allow for two inputs

    def __init__(self):
        super().__init__()

        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)
        # block encoder
        self.multiresblock1 = multiresblock(inchannel=1, outchannelofmultiresblock=64, firstblock=True)
        self.respath1 = respath(inchannel=128, outchannel=64, depth=3)

        self.multiresblock2 = multiresblock(inchannel=64, outchannelofmultiresblock=128, firstblock=False)
        self.respath2 = respath(inchannel=256, outchannel=128, depth=2)

        self.multiresblock3 = multiresblock(inchannel=128, outchannelofmultiresblock=256, firstblock=False)
        self.respath3 = respath(inchannel=512, outchannel=256, depth=1)

        # bridge
        self.multiresblock4 = multiresblock(inchannel=512, outchannelofmultiresblock=256, firstblock=False)
        self.upsample1 = upsampling(256, 256)

        # decoder
        self.multiresblock5 = multiresblock(inchannel=512, outchannelofmultiresblock=128, firstblock=False)
        self.upsample2 = upsampling(128, 128)

        self.multiresblock6 = multiresblock(inchannel=256, outchannelofmultiresblock=64, firstblock=False)
        self.upsample3 = upsampling(64, 64)

        self.multiresblock7 = multiresblock(inchannel=128, outchannelofmultiresblock=64, firstblock=False)
        self.upsample_b6 = upsampling(64, 64)

        # last layer
        # last layer
        self.last_layer = nn.Sequential(
            batchnorm_and_relu(64),
            nn.Conv3d(in_channels=64, out_channels=9, kernel_size=3, padding='same'))

    ######multiresunetwithmultiresblocks###################
    def forward(self, input1, input2):
        # encoding
        skip1_1 = self.multiresblock1(input1)  # 4,64,144,144,22
        output1_1 = self.down(skip1_1)  # 4,64,72,72,21

        skip1_2 = self.multiresblock1(input2)  # 4,64,144,144,22
        output1_2 = self.down(skip1_2)  # 4,64,72,72,21

        respathinput1 = torch.cat((skip1_1, skip1_2), 1)  # 4,128,144,144,22
        respathskip1 = self.respath1(respathinput1)  # 4,64,144,144,22

        skip2_1 = self.multiresblock2(output1_1)  # 4,128,72,72,21
        output2_1 = self.down(skip2_1)  # 4,128,36,36,20

        skip2_2 = self.multiresblock2(output1_2)  # 4,128,72,72,21
        output2_2 = self.down(skip2_2)  # 4,128,36,36,20

        respathinput2 = torch.cat((skip2_1, skip2_2), 1)
        respathskip2 = self.respath2(respathinput2)

        skip3_1 = self.multiresblock3(output2_1)  # 4,256,36,36,20
        output3_1 = self.down(skip3_1)  # 4,256,18,18,19

        skip3_2 = self.multiresblock3(output2_2)  # 4,256,36,36,20
        output3_2 = self.down(skip3_2)  # 4,256,18,18,19

        respathinput3 = torch.cat((skip3_1, skip3_2), 1)
        respathskip3 = self.respath3(respathinput3)

        ##########bridge (b4)#########
        concatencoder = torch.cat((output3_1, output3_2), 1)  # 4,512,18,18,19
        bridge = self.multiresblock4(concatencoder)  # 4,256,18,18,19
        # upconvolution bridge (b4)
        out_bridge = self.upsample1(bridge)  # 4,256,36,36,20
        ###########################

        # concat bridge and encoder skips
        inputb5 = torch.cat((out_bridge, respathskip3), 1)  # 4,512,36,36,20
        b5 = self.multiresblock5(inputb5)  # 4,128,36,36,20
        output5 = self.upsample2(b5)  # 4,128,72,72,21

        # decoding block6
        inputb6 = torch.cat((output5, respathskip2), 1)  # 4,256,72,72,21
        b6 = self.multiresblock6(inputb6)  # 4,64,72,72,21
        output6 = self.upsample3(b6)  # 4,64,144,144,22

        # decoding block7
        inputb7 = torch.cat((output6, respathskip1), 1)  # 4,128,144,144,22
        b7 = self.multiresblock7(inputb7)  # 4,64,144,144,22

        x = self.last_layer(b7)  # 4,9,144,144,22
        output = F.softmax(x, dim=1)
        return output


######################################
########multikultiunet############
######################################

class multikultiunet(nn.Module):
    def __init__(self):
        super().__init__()
        """Initialize neural net layers."""
        self.firstlayerdensblock1 = nn.Conv3d(in_channels=1, out_channels=13, kernel_size=3, padding='same')

        self.secondlayerdensblock1 = nn.Sequential(
            batchnorm_and_relu(14),
            nn.Conv3d(in_channels=14, out_channels=26, kernel_size=3, padding='same')
        )
        self.comp1 = transitiondown(40, 32)
        self.down = nn.MaxPool3d(kernel_size=2, stride=[2, 2, 1], padding=0)

        # block encoder
        self.densblock2 = densblock(filters=12, repetitions=2, inchannel=32)
        self.comp2 = transitiondown(164, 64)

        self.densblock3 = densblock(filters=12, repetitions=2, inchannel=64)
        self.comp3 = transitiondown(292, 128)

        # blocks decoder

        self.resblock1 = resblock(256, 256)  # bridge
        self.upsample1 = upsampling(256, 256)

        self.resblock2 = resblock(512, 128)
        self.upsample2 = upsampling(128, 128)

        self.resblock3 = resblock(256, 64)
        self.upsample3 = upsampling(64, 64)

        self.resblock4 = resblock(128, 64)
        self.upsample4 = upsampling(64, 64)

        # last layer
        # last layer
        self.last_layer = nn.Sequential(
            batchnorm_and_relu(64),
            nn.Conv3d(in_channels=64, out_channels=9, kernel_size=3, padding='same'))

    ######multikultiunet alt###################
    def forward(self, input1, input2):
        # encoding
        # densblock1.1
        first1 = self.firstlayerdensblock1(input1)  # 4,13,144,144,22
        concatinputandfirst1 = torch.cat((first1, input1), 1)  # 4,14,144,144,22
        second1 = self.secondlayerdensblock1(concatinputandfirst1)  # 4,26,144,144,22
        encoderconcat1_1 = torch.cat((second1, first1, input1), 1)  # 4,40,144,144,22

        skip1_1 = self.comp1(encoderconcat1_1)  # 4,32,144,144,22
        # maxpooling 1_1
        output1_1 = self.down(skip1_1)  # 4,32,72,72,21

        # densblock1.2
        first2 = self.firstlayerdensblock1(input2)  # 4,13,144,144,22
        concatinputandfirst2 = torch.cat((first2, input2), 1)  # 4,14,144,144,22
        second2 = self.secondlayerdensblock1(concatinputandfirst2)  # 4,26,144,144,22
        encoderconcat1_2 = torch.cat((second2, first1, input2), 1)  # 4,40,144,144,22
        skip1_2 = self.comp1(encoderconcat1_2)  # 4,32,144,144,22
        # maxpooling 2.1
        output1_2 = self.down(skip1_2)  # 4,32,72,72,21

        # densblock 2.1
        db2_1 = self.densblock2(output1_1)  # densblock out: 4,132,72,72,21
        encoderconcat2_1 = torch.cat((db2_1, output1_1),
                                     1)  # 4,164,72,72,21 #concatinates input and output of densblock
        skip2_1 = self.comp2(encoderconcat2_1)  # 4,64,72,72,21
        # maxpooling 2.1
        output2_1 = self.down(skip2_1)  # 4,64,36,36,20 maxpooling

        # densblock 2.2
        db2_2 = self.densblock2(output1_2)  # densblock out: 4,132,72,72,21
        encoderconcat2_2 = torch.cat((db2_2, output1_2),
                                     1)  # 4,164,72,72,21 #concatinates input and output of densblock
        skip2_2 = self.comp2(encoderconcat2_2)  # 4,64,144,144,22
        # maxpooling 2.1
        output2_2 = self.down(skip2_2)  # 4,64,36,36,20 maxpooling

        # densblock 3.1
        db3_1 = self.densblock3(output2_1)  # densblock out: #4,228,36,36,20
        encoderconcat3_1 = torch.cat((db3_1, output2_1),
                                     1)  # 4,292,36,36,20 #concatinates input and output of densblock
        skip3_1 = self.comp3(encoderconcat3_1)  # 4,128,36,36,20
        # maxpooling 3.1
        output3_1 = self.down(skip3_1)  # 4,128,18,18,19 maxpooling

        # densblock 3.2
        db3_2 = self.densblock3(output2_2)  # densblock out: #4,228,36,36,20
        encoderconcat3_2 = torch.cat((db3_2, output2_2),
                                     1)  # 4,292,36,36,20 #concatinates input and output of densblock
        skip3_2 = self.comp3(encoderconcat3_2)  # 4,128,36,36,20
        # maxpooling 3.2
        output3_2 = self.down(skip3_2)  # 4,128,18,18,19 maxpooling

        ##########bridge (b4)#########
        concatencoder = torch.cat((output3_1, output3_2), 1)  # 4,256,18,18,19
        bridge_rb1 = self.resblock1(concatencoder)  # 4,256,18,18,19
        # upconvolution bridge (b4)
        out_bridge = self.upsample1(bridge_rb1)  # 4,256,36,36,20
        ###########################

        # concat bridge and encoder skips
        inputrb2 = torch.cat((out_bridge, skip3_1, skip3_2), 1)  # 4,512,36,36,20

        # resnetblock 2
        rb2 = self.resblock2(inputrb2)  # 4,128,36,36,20
        outputrb2 = self.upsample2(rb2)  # 4,128,72,72,21
        inputrb3 = torch.cat((outputrb2, skip2_1, skip2_2), 1)  # 4,256,72,72,21

        # resnetblock 3
        rb3 = self.resblock3(inputrb3)  # 4,64,72,72,21
        outputrb3 = self.upsample3(rb3)  # 4,64,144,144,22
        inputrb4 = torch.cat((outputrb3, skip1_1, skip1_2), 1)  # 4,128,144,144,22

        # resnetblock 4
        rb4 = self.resblock4(inputrb4)  # 4,64,144,144,22
        x = self.last_layer(rb4)  # 4,9,144,144,22

        output = F.softmax(x, dim=1)
        return output


class hoffentlich_alex(nn.Module):
    # this is my own brainchild, it is not verified and not published,
    # i came up with this while looking at: https://doi.org/10.1145%2F3065386
    # but even a brief comparison will tell you that the two models have nothing in common
    # use this with extreme precaution, we have seen good results for our tasks
    # but again: not published, not peer reviewed

    def __init__(self, numberlabels):
        super(hoffentlich_alex, self).__init__()
        """Initialize neural net layers."""
        self.dropout_rate = 0.01
        self.First_Convolution = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), padding='same').to('cuda:0'),
            ELU(),

        )
        self.Average_pooling_block_1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 1), padding=0, ceil_mode=True).to('cuda:0'), ELU())
        self.DownsamplingSQE_1_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=8, out_channels=32, padding='same').to('cuda:0'),
            ELU(),
            Dropout3d(p=self.dropout_rate).to('cuda:0'),
            BatchNorm3d(num_features=32).to('cuda:0'),
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=32, out_channels=32, padding='same').to('cuda:0'),
            ELU(),
            Dropout3d(p=self.dropout_rate).to('cuda:0'),
            BatchNorm3d(num_features=32).to('cuda:0')
        )
        self.DownsamplingSQE_1_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=40, out_channels=32, padding='same').to('cuda:0'),
            ELU().to('cuda:0'),
            Dropout3d(p=self.dropout_rate).to('cuda:0'),
            BatchNorm3d(num_features=32).to('cuda:0'),
        )
        self.Intermediate_SQE_1_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=32, out_channels=64, padding='same').to('cuda:0'),
            ELU().to('cuda:0'),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same').to('cuda:0'),
            ELU().to('cuda:0'),
            Dropout3d(p=self.dropout_rate).to('cuda:0'),
            BatchNorm3d(num_features=64).to('cuda:0'),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same').to('cuda:0'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_1_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=96, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )

        self.Average_pooling_block_2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 1), ceil_mode=True), ELU())
        self.DownsamplingSQE_2_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.DownsamplingSQE_2_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_2_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_2_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Average_pooling_block_3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 1), padding=0, ceil_mode=True), ELU())
        self.DownsamplingSQE_3_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.DownsamplingSQE_3_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_3_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_3_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Upsampling_Block_1 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 1)), ELU())
        self.UpsamplingSQE_1_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.UpsamplingSQE_1_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=192, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_4_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_4_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Upsampling_Block_2 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 1)), ELU())
        self.UpsamplingSQE_2_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.UpsamplingSQE_2_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=160, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_5_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_5_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Upsampling_Block_3 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 1)), ELU())
        self.UpsamplingSQE_3_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.UpsamplingSQE_3_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=129, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_6_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_6_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.LastConvolution_Layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU()
        )
        self.output_layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=numberlabels, padding='same'),
            Softmax()
        )

    def forward(self, x):
        data = x
        x1 = self.First_Convolution(x)
        x1 = x1.to('cuda:0')
        Average_pooling_1 = self.Average_pooling_block_1(x1)
        Average_pooling_1 = Average_pooling_1.to('cuda:0')
        x2 = self.DownsamplingSQE_1_part_1(Average_pooling_1)
        x2 = x2.to('cuda:0')
        x3 = cat((Average_pooling_1, x2), 1)
        x3 = self.DownsamplingSQE_1_part_2(x3)  # hierher_concatinieren
        x3 = x3.to('cuda:0')
        x4 = self.Intermediate_SQE_1_part_1(x3)
        x4 = x4.to('cuda:0')
        x4 = cat((x3, x4), 1)
        x4 = self.Intermediate_SQE_1_part_2(x4)
        x4 = x4.to('cuda:0')
        x4 = self.Average_pooling_block_2(x4)
        x4 = x4.to('cuda:0')
        x5 = self.DownsamplingSQE_2_part_1(x4)
        x5 = x5.to('cuda:0')
        x5 = cat((x4, x5), 1)
        x6 = self.DownsamplingSQE_2_part_2(x5)  # hierher_concatinieren
        x6 = x6.to('cuda:0')
        x7 = self.Intermediate_SQE_2_part_1(x6)
        x7 = cat((x6, x7), 1)
        x7 = self.Intermediate_SQE_2_part_2(x7)

        x8 = self.Average_pooling_block_3(x7)
        x9 = self.DownsamplingSQE_3_part_1(x8)
        x9 = cat((x8, x9), 1)
        x10 = self.DownsamplingSQE_3_part_2(x9)

        x11 = self.Intermediate_SQE_3_part_1(x10)
        x10 = cat((x10, x11), 1)
        x10 = self.Intermediate_SQE_3_part_2(x10)

        x11 = self.Upsampling_Block_1(x10)
        x12 = self.UpsamplingSQE_1_part_1(x11)
        x12 = cat((x6, x11, x12), 1)
        x12 = self.UpsamplingSQE_1_part_2(x12)

        x13 = self.Intermediate_SQE_4_part_1(x12)
        x13 = cat((x12, x13), 1)
        x13 = self.Intermediate_SQE_4_part_2(x13)

        x13 = self.Upsampling_Block_2(x13)
        x14 = self.UpsamplingSQE_2_part_1(x13)
        x14 = cat((x3, x13, x14), 1)
        x14 = self.UpsamplingSQE_2_part_2(x14)

        x15 = self.Intermediate_SQE_5_part_1(x14)
        x15 = cat((x14, x15), 1)
        x15 = self.Intermediate_SQE_5_part_2(x15)

        x15 = self.Upsampling_Block_3(x15)
        x16 = self.UpsamplingSQE_3_part_1(x15)
        x16 = cat((data, x15, x16), 1)
        x16 = self.UpsamplingSQE_3_part_2(x16)
        # current
        x17 = self.Intermediate_SQE_6_part_1(x16)
        x17 = cat((x16, x17), 1)
        x17 = self.Intermediate_SQE_5_part_2(x17)

        x17 = self.LastConvolution_Layer(x17)
        x17 = self.output_layer(x17)
        return x17


class hoffentlich_alex_v2(nn.Module):
    # this is the second version of my own brainchild, it is not verified and not published,
    # i came up with this while looking at: https://doi.org/10.1145%2F3065386
    # but even a brief comparison will tell you that the two models have nothing in common
    # use this with extreme precaution, we have seen good results for our tasks
    # but again: not published, not peer reviewed
    def __init__(self, numberlabels):
        super(hoffentlich_alex_v2, self).__init__()
        """Initialize neural net layers."""
        self.dropout_rate = 0.01
        self.First_Convolution = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), padding='same'),
            ELU(),

        )
        self.Average_pooling_block_1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 1), padding=0, ceil_mode=True, return_indices=True))
        self.DownsamplingSQE_1_part_1 = nn.Sequential(
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=8, out_channels=32, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=32),
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=32, out_channels=32, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=32)
        )
        self.DownsamplingSQE_1_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3), in_channels=40, out_channels=32, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=32),
        )
        self.Intermediate_SQE_1_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=32, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_1_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=96, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )

        self.Average_pooling_block_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 1), padding=0, ceil_mode=True, return_indices=True))
        self.DownsamplingSQE_2_part_1 = nn.Sequential(
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.DownsamplingSQE_2_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_2_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_2_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Average_pooling_block_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(2, 2, 1), padding=0, ceil_mode=True, return_indices=True))
        self.DownsamplingSQE_3_part_1 = nn.Sequential(
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.DownsamplingSQE_3_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_3_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_3_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Upsampling_Block_1 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 1)), ELU())
        self.UpsamplingSQE_1_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.UpsamplingSQE_1_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=192, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_4_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_4_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Upsampling_Block_2 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 1)), ELU())
        self.UpsamplingSQE_2_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.UpsamplingSQE_2_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=160, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_5_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_5_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Upsampling_Block_3 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 1)), ELU())
        self.UpsamplingSQE_3_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.UpsamplingSQE_3_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=129, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
        )
        self.Intermediate_SQE_6_part_1 = nn.Sequential(
            nn.Conv3d(kernel_size=(1, 1, 1), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64),
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.Intermediate_SQE_6_part_2 = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=128, out_channels=64, padding='same'),
            ELU(),
            Dropout3d(p=self.dropout_rate),
            BatchNorm3d(num_features=64)
        )
        self.LastConvolution_Layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU()
        )
        self.output_layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=numberlabels, padding='same'),
            Softmax(dim=1)
        )

    def forward(self, x):
        data = x
        x1 = self.First_Convolution(x)
        Average_pooling_1 = self.Average_pooling_block_1(x1)
        x2 = self.DownsamplingSQE_1_part_1(Average_pooling_1)
        x3 = cat((Average_pooling_1, x2), 1)
        x3 = self.DownsamplingSQE_1_part_2(x3)  # hierher_concatinieren

        x4 = self.Intermediate_SQE_1_part_1(x3)
        x4 = cat((x3, x4), 1)
        x4 = self.Intermediate_SQE_1_part_2(x4)

        x4 = self.Average_pooling_block_2(x4)
        x5 = self.DownsamplingSQE_2_part_1(x4)
        x5 = cat((x4, x5), 1)
        x6 = self.DownsamplingSQE_2_part_2(x5)  # hierher_concatinieren

        x7 = self.Intermediate_SQE_2_part_1(x6)
        x7 = cat((x6, x7), 1)
        x7 = self.Intermediate_SQE_2_part_2(x7)

        x8 = self.Average_pooling_block_3(x7)
        x9 = self.DownsamplingSQE_3_part_1(x8)
        x9 = cat((x8, x9), 1)
        x10 = self.DownsamplingSQE_3_part_2(x9)

        x11 = self.Intermediate_SQE_3_part_1(x10)
        x10 = cat((x10, x11), 1)
        x10 = self.Intermediate_SQE_3_part_2(x10)

        x11 = self.Upsampling_Block_1(x10)
        x12 = self.UpsamplingSQE_1_part_1(x11)
        x12 = cat((x6, x11, x12), 1)
        x12 = self.UpsamplingSQE_1_part_2(x12)

        x13 = self.Intermediate_SQE_4_part_1(x12)
        x13 = cat((x12, x13), 1)
        x13 = self.Intermediate_SQE_4_part_2(x13)

        x13 = self.Upsampling_Block_2(x13)
        x14 = self.UpsamplingSQE_2_part_1(x13)
        x14 = cat((x3, x13, x14), 1)
        x14 = self.UpsamplingSQE_2_part_2(x14)

        x15 = self.Intermediate_SQE_5_part_1(x14)
        x15 = cat((x14, x15), 1)
        x15 = self.Intermediate_SQE_5_part_2(x15)

        x15 = self.Upsampling_Block_3(x15)
        x16 = self.UpsamplingSQE_3_part_1(x15)
        x16 = cat((data, x15, x16), 1)
        x16 = self.UpsamplingSQE_3_part_2(x16)
        # current
        x17 = self.Intermediate_SQE_6_part_1(x16)
        x17 = cat((x16, x17), 1)
        x17 = self.Intermediate_SQE_5_part_2(x17)

        x17 = self.LastConvolution_Layer(x17)
        x17 = self.output_layer(x17)
        return x17


class hoffentlich_alex_with_modules(nn.Module):
    # this is the second version of my own brainchild, it is not verified and not published,
    # i came up with this while looking at: https://doi.org/10.1145%2F3065386
    # but even a brief comparison will tell you that the two models have nothing in common
    # use this with extreme precaution, we have seen good results for our tasks
    # but again: not published, not peer reviewed
    def __init__(self, numberlabels):
        super(hoffentlich_alex_with_modules, self).__init__()
        """Initialize neural net layers."""
        self.dropout_rate = 0.01
        self.First_Convolution = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=100, kernel_size=(3, 3, 3), padding='same'),
            ELU()).to('cuda:0')
        self.dropout_rate = 0.1
        self.Downsampling_1 = Alex_Downsampling_Block(100, 100, 100, self.dropout_rate).to(
            'cuda:0')  # (self,in_channels, channels_previous_layers, internal_channels,dropout_rate):
        self.Intermediate_1 = Alex_Intermediate_Block(100, 100, 100, self.dropout_rate).to('cuda:0')
        self.Downsampling_2 = Alex_Downsampling_Block(100, 100, 100, self.dropout_rate).to('cuda:0')
        self.Intermediate_2 = Alex_Intermediate_Block(100, 100, 100, self.dropout_rate).to('cuda:0')
        self.Downsampling_3 = Alex_Downsampling_Block(100, 100, 100, self.dropout_rate).to('cuda:0')
        self.Intermediate_3 = Alex_Intermediate_Block(100, 100, 100, self.dropout_rate).to('cuda:1')
        self.Upsampling_1 = Alex_Upsampling_Block(100, 100, 100, 100, self.dropout_rate).to(
            'cuda:1')  # (self,in_channels, channels_previous_layers, channels_downsampling_output, internal_channels,dropout_rate):
        self.Intermediate_4 = Alex_Intermediate_Block(100, 100, 100, self.dropout_rate).to('cuda:1')
        self.Upsampling_2 = Alex_Upsampling_Block(100, 100, 100, 100, self.dropout_rate).to('cuda:1')
        self.Intermediate_5 = Alex_Intermediate_Block(100, 100, 100, self.dropout_rate).to('cuda:1')
        self.Upsampling_3 = Alex_Upsampling_Block(100, 100, 100, 100, self.dropout_rate).to('cuda:1')
        self.LastConvolution_Layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=100, out_channels=64, padding='same'),
            ELU()
        ).to('cuda:1')
        self.output_layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=numberlabels + 1, padding='same'),
            Softmax(dim=1)
        ).to('cuda:1')

    def forward(self, x):
        x = x.to('cuda:0')
        data = self.First_Convolution(x)

        x1 = self.Downsampling_1(data)
        x2 = self.Intermediate_1(x1)
        x3 = self.Downsampling_2(x2)
        x4 = self.Intermediate_2(x3)
        x5 = self.Downsampling_3(x4)
        x5 = x5.to('cuda:1')
        x6 = self.Intermediate_3(x5)
        # x6 = x6.to('cuda:1')
        x4 = x4.to('cuda:1')
        x7 = self.Upsampling_1(x6, x4)
        x8 = self.Intermediate_4(x7)
        x2 = x2.to('cuda:1')
        x9 = self.Upsampling_2(x8, x2)
        x10 = self.Intermediate_5(x9)
        data = data.to('cuda:1')
        x10 = self.Upsampling_3(x10, data)
        x10 = self.LastConvolution_Layer(x10)
        x10 = self.output_layer(x10)
        x10 = x10.to('cuda:0')
        return x10


class Hoffentlich_Alex_with_Modules_with_fancy_upsampling(nn.Module):
    # this is the second version of my own brainchild, it is not verified and not published,
    # i came up with this while looking at: https://doi.org/10.1145%2F3065386
    # but even a brief comparison will tell you that the two models have nothing in common
    # use this with extreme precaution, we have seen good results for our tasks
    # but again: not published, not peer reviewed
    def __init__(self):
        super(Hoffentlich_Alex_with_Modules_with_fancy_upsampling, self).__init__()
        """Initialize neural net layers."""
        self.dropout_rate = 0.01
        self.First_Convolution = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), padding='same'),
            ELU())
        self.dropout_rate = 0.05
        self.First_Convolution = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 3, 3), padding='same'),
            ELU())
        self.dropout_rate = 0.05
        self.Downsampling_1 = Alex_Downsampling_Block_v2(64, 64, 64, self.dropout_rate)
        self.Intermediate_1 = Alex_Intermediate_Block(64, 64, 64, self.dropout_rate)
        self.Downsampling_2 = Alex_Downsampling_Block_v2(64, 64, 64, self.dropout_rate)
        self.Intermediate_2 = Alex_Intermediate_Block(64, 64, 64, self.dropout_rate)
        self.Downsampling_3 = Alex_Downsampling_Block_v2(64, 64, 64, self.dropout_rate)
        self.Intermediate_3 = Alex_Intermediate_Block(64, 64, 64, self.dropout_rate)
        self.Upsampling_1 = Alex_Upsampling_Block_v2(64, 64, 64, 64, self.dropout_rate)
        self.Intermediate_4 = Alex_Intermediate_Block(64, 64, 64, self.dropout_rate)
        self.Upsampling_2 = Alex_Upsampling_Block_v2(64, 64, 64, 64, self.dropout_rate)
        self.Intermediate_5 = Alex_Intermediate_Block(64, 64, 64, self.dropout_rate)
        self.Upsampling_3 = Alex_Upsampling_Block_v2(64, 64, 64, 64, self.dropout_rate)
        self.LastConvolution_Layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=64, padding='same'),
            ELU()
        )
        self.output_layer = nn.Sequential(
            nn.Conv3d(kernel_size=(3, 3, 3,), in_channels=64, out_channels=9, padding='same'),
            Softmax(dim=1)
            # only do softmax once: Whithe each apply to softmax we get less separation -> the training algorithm will improve more slowly.
        )

    def forward(self, x):
        data = self.First_Convolution(x)
        x1, indices1 = self.Downsampling_1(data)
        x2 = self.Intermediate_1(x1)
        x3, indices2 = self.Downsampling_2(x2)
        x4 = self.Intermediate_2(x3)
        x5, indices3 = self.Downsampling_3(x4)
        x6 = self.Intermediate_3(x5)
        x7 = self.Upsampling_1(x6, x4, indices3)
        x8 = self.Intermediate_4(x7)
        x9 = self.Upsampling_2(x8, x2, indices2)
        x10 = self.Intermediate_5(x9)
        x10 = self.Upsampling_3(x10, data, indices1)
        x10 = self.LastConvolution_Layer(x10)
        x10 = self.output_layer(x10)
        return x10
