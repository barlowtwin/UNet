import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

	def __init__(self, in_channels, out_channels, mid_channels = None):
		super(DoubleConv, self).__init__()

		if not mid_channels :
			mid_channels = out_channels

		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace = True),
			nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace = True))

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	# max pool followed by doubleConv

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(kernel_size = 2),
			DoubleConv(in_channels, out_channels))

	def forward(self, x):
		return self.maxpool_conv(x)



class Up(nn.Module):

	# upscale followed by DoubleConv

	def __init__(self, in_channels, out_channels, bilinear = True):
		super(Up, self).__init__()

		if bilinear :
			self.up = nn.Upsample(scale_factor = 2, model = 'bilinear', align_corners = True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels//2)

		else :
			self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
			self.conv = DoubleConv(in_channels, out_channels)


	def forward(self, x1 , x2):

		# there might be slight differences in x and y dim of channels
		# to make up for it we use padding before concatenation

		x1 = self.up(x1)

		diffy = x2.size()[2] - x1.size()[2]
		diffx = x2.size()[3] - x1.size()[3]

		# left, right, top, bottom
		x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2])
		x = torch.cat([x1, x2], dim = 1)
		return self.conv(x)




class OutConv(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

	def forward(self, x):
		return self.conv(x)
