import torch
import torch.nn as nn
from concern.charsets import DefaultCharset


class SegRecognizer(nn.Module):
    def __init__(self, in_channels, charset=DefaultCharset(), inner_channels=256, use_resnet=False, bias=False):
        super(SegRecognizer, self).__init__()
        self.use_resnet = use_resnet
        self.mask = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
                nn.Conv2d(inner_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid())
        self.classify = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
                nn.Conv2d(inner_channels, len(charset), kernel_size=1, padding=0))
    
        #for fpn
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0) #to reduce channels
        #upsample 
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearset')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearset')

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return nn.functional.interpolate(x, size=(H,W), mode='bilinear') + y



    def forward(self, feature, mask=None, classify=None, train=False):
        '''
        Args:
            feature: Features extracted from backbone with shape N, C, H, W.
            mask: N, H, W. Float tensor indicating the text regions.
            classify: N, H, W. Integer tensor indicating the text classes.
        '''
        cur_feature = feature

        if self.use_resnet:
            x2, x3, x4, x5 = feature
            p5 = self.toplayer(x5)
            p4 = self._upsample_add(p5, self.latlayer1(x4)) #self.up5(p5) + self.latlayer1(x4)
            p3 = self._upsample_add(p4, self.latlayer2(x3)) #self.up4(p4) + self.latlayer2(x3)
            p2 = self._upsample_add(p3, self.latlayer3(x2)) #self.up3(p3) + self.latlayer3(x2)
            
            #Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            p2 = self.smooth3(p2)

            cur_feature = p2

        mask_pred = self.mask(cur_feature)
        mask_pred = mask_pred.squeeze(1)  # N, H, W
        classify_pred = self.classify(cur_feature)
        pred = dict(mask=mask_pred, classify=classify_pred)
        if self.training:
            mask_loss = nn.functional.binary_cross_entropy(mask_pred, mask)
            classify_loss = nn.functional.cross_entropy(classify_pred, classify)
            loss = mask_loss + classify_loss
            metrics = dict(mask_loss=mask_loss, classify_loss=classify_loss)
            return loss, pred, metrics
        pred['classify'] = nn.functional.softmax(classify_pred, dim=1)
        return pred
