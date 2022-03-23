from utils import DoubleConv, Up, Down, OutConv
import torch.nn as nn

class Unet(nn.Module):

	def __init__(self, in_channels, num_classes, bilinear = False):
		super(Unet, self).__init__()

		self.in_channels = in_channels
		self.num_classes = num_classes
		self.bilinear = bilinear

		self.start = DoubleConv(in_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)

		factor = 2 if bilinear else 1 # because we use ConvT otherwise
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.out = OutConv(64, num_classes)



	def forward(self, x):

		x1 = self.start(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.out(x)
		return logits