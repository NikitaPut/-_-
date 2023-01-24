# # from https://github.com/amdegroot/ssd.pytorch


# import torch
# from torchvision import transforms
# import cv2
# import numpy as np
# import types
# from numpy import random
# from PIL import Image
# import time

# class Compose(object):
#     """Composes several augmentations together.
#     Args:
#         transforms (List[Transform]): list of transforms to compose.
#     Example:
#         >>> augmentations.Compose([
#         >>>     transforms.CenterCrop(10),
#         >>>     transforms.ToTensor(),
#         >>> ])
#     """

#     def __init__(self, transforms):
#         self.transforms = transforms
        
#         self.transform_time = None
#         self.counter = 0

#     def __call__(self, imgs, labels=None, mask=None):
        
#         tr_times = []
#         for t in self.transforms:
#             t0 = time.time()
#             imgs, labels, mask = t(imgs, labels, mask)
#             t1 = time.time()

#             tr_times.append(t1 - t0)
        
#         self.transform_time = np.array(tr_times)
#         self.counter += 1
        
# #         np.set_printoptions(precision=3)
# #         print("Tr: %s %.2f" %(self.transform_time*1000, np.sum(self.transform_time)*1000))

#         return imgs, labels, mask

# class Normalize(object):
#     def __call__(self, images, labels=None, mask = None):
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         for i in range(0, len(images)):
#             # print(images[i].shape)
#             images[i][:,:,0] = (images[i][:,:,0] - mean[2]) / std[2]
#             images[i][:,:,1] = (images[i][:,:,1] - mean[1]) / std[1]
#             images[i][:,:,2] = (images[i][:,:,2] - mean[0]) / std[0]
#         return images, labels, mask

# class ConvertFromInts(object):
#     def __call__(self, images, labels=None, mask = None):
#         return images.astype(np.float32), labels.astype(np.float32), mask

# class SubtractMeans(object):
#     def __init__(self, mean):
#         self.mean = np.array(mean, dtype=np.float32)

#     def __call__(self, image, boxes=None, labels=None):
#         image = image.astype(np.float32)
#         image -= self.mean
#         return image.astype(np.float32), boxes, labels

# class ConvertFromInts(object):
#     def __call__(self, images, labels=None, mask = None):
#         for i in range(0, len(images)):
#             images[i] = images[i].astype(np.float32)
#         return images, labels, mask

# class ConvertToInts(object):
#     def __call__(self, images, labels=None, mask = None):
#         for i in range(0, len(images)):
#             images[i] = images[i].astype(np.uint8)
#         return images, labels, mask

# # class RandomTransform(object):
# #     def get_transform(self, src_size, dst_size): 
# #         
# #         pass
# #     def __call__(self, images, labels=None, mask = None):
# #         if random.randint(2):
# #             min_crop = 0.5
# #             scale_x = min(min_crop + random.random() * (1.0 - min_crop), 1.0)
# #             scale_y = min(min_crop + random.random() * (1.0 - min_crop), 1.0)
# #             new_h = int(images[0].shape[0] * scale_y)
# #             new_w = int(images[0].shape[1] * scale_x)
# # 
# #             shift_y = int(random.random() * (images[0].shape[0] - new_h))
# #             shift_x = int(random.random() * (images[0].shape[1] - new_w))
# #             
# # #             print("RandomCrop: ", scale_x, scale_y, new_h, new_w, shift_y, shift_x)
# # 
# #             for i in range(0, len(images)):
# #                 images[i] = images[i][shift_y:new_h + shift_y, shift_x:new_w + shift_x, :]
# #             labels = labels[shift_y:new_h + shift_y, shift_x:new_w + shift_x]
# #             mask = mask[shift_y:new_h + shift_y, shift_x:new_w + shift_x]
# #         return images, labels, mask


# class RandomCrop(object):
#     def __call__(self, images, labels=None, mask = None):
#         if random.randint(2):
#             min_crop = 0.95 # TODO: 0.6 was
#             scale_x = min(min_crop + random.random() * (1.0 - min_crop), 1.0)
#             scale_y = min(min_crop + random.random() * (1.0 - min_crop), 1.0)
#             new_h = int(images[0].shape[0] * scale_y)
#             new_w = int(images[0].shape[1] * scale_x)

#             shift_y = int(random.random() * (images[0].shape[0] - new_h))
#             shift_x = int(random.random() * (images[0].shape[1] - new_w))
            
# #             print("RandomCrop: ", scale_x, scale_y, new_h, new_w, shift_y, shift_x)

#             for i in range(0, len(images)):
#                 images[i] = images[i][shift_y:new_h + shift_y, shift_x:new_w + shift_x, :]
#             labels = labels[shift_y:new_h + shift_y, shift_x:new_w + shift_x]
#             mask = mask[shift_y:new_h + shift_y, shift_x:new_w + shift_x]
#         return images, labels, mask

# class RandomJpeg(object):
#     def __call__(self, images, labels=None, mask = None):
#         if random.randint(2):
#             encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
#             for i in range(0, len(images)):
#                 result, encimg = cv2.imencode('.jpg', images[i], encode_param)
#                 images[i] = cv2.imdecode(encimg, 1)
#         return images, labels, mask

# class RandomMirror(object):
#     def __call__(self, images, labels=None, mask = None):
#         do_flip = random.randint(2)
#         is_hori  = random.randint(2)
#         if do_flip:
#             for i in range(0, len(images)):
#                 _, width, _ = images[i].shape
#                 if is_hori:
#                     images[i] = images[i][:, ::-1]
#                     labels = labels[:, ::-1]
#                     mask = mask[:,::-1]
#                 else:
#                     images[i] = images[i][::-1]
#                     labels = labels[::-1]
#                     mask = mask[::-1]
#         return images, labels, mask

# class RandomShift(object):
#     def __init__(self, distance = 0.1):
#         self.distance = distance

#     def __call__(self, images, labels=None, mask=None):
#         if random.randint(2):
#             axis = random.randint(2)
#             shift_size = int(images[0].shape[0] * self.distance)

#             if random.randint(2):
#                 start = 0
#                 end = shift_size
#             else:
#                 start = shift_size
#                 end = 0
#             for i in range(0, len(images)):
#                 im = images[i]
#                 if axis == 0:
#                     images[i] = np.pad(im,((0,0),(start,end),(0,0)), mode='constant', constant_values = 255.)[:, end:im.shape[1] + end, :]
#                     # labels = np.pad(labels,((0,0),(start,end)), mode='constant', constant_values = 0.)[:, end:labels.shape[1] + end]
#                 else:
#                     images[i] = np.pad(im,((start,end),(0,0),(0,0)), mode='constant', constant_values = 255.)[end:im.shape[0] + end :, :]
#                     # labels = np.pad(labels,((start,end),(0,0)), mode='constant', constant_values = 0.)[end:labels.shape[0] + end :]

#             #shift mask
#             if axis == 0:
#                 mask = np.pad(mask,((0,0),(start,end)), mode='constant', constant_values = 255.)[:,  end:im.shape[1] + end]
#                 labels = np.pad(labels,((0,0),(start,end)), mode='constant', constant_values = 0.)[:, end:labels.shape[1] + end]
#             else:
#                 mask = np.pad(mask,((start,end),(0,0)), mode='constant', constant_values = 255.)[end:im.shape[0] + end, :]
#                 labels = np.pad(labels,((start,end),(0,0)), mode='constant', constant_values = 0.)[end:labels.shape[0] + end :]
#         return images, labels, mask

# class RandomContrast(object):
#     def __init__(self, lower=0.75, upper=1.25):
#         self.lower = lower
#         self.upper = upper
#         assert self.upper >= self.lower, "contrast upper must be >= lower."
#         assert self.lower >= 0, "contrast lower must be non-negative."

#     # expects float image
#     def __call__(self, images, labels=None, mask=None):
#         if random.randint(2):
#             alpha = random.uniform(self.lower, self.upper)
#             for i in range(0, len(images)):
#                 images[i] *= alpha
#         return images, labels, mask

# class RandomGamma(object):
#     def __init__(self, lower=0.5, upper=2.0):
#         self.lower = lower
#         self.upper = upper
#         assert self.upper >= self.lower, "contrast upper must be >= lower."
#         assert self.lower >= 0, "contrast lower must be non-negative."

#     # expects float image
#     def __call__(self, images, labels=None, mask=None):
#         if random.randint(2):
#             gamma = random.uniform(self.lower, self.upper)
#             if np.mean(images[0]) > 100:
#                 for i in range(0, len(images)):
#                     images[i] = pow(images[i] / 255., gamma) * 255.
#         return images, labels, mask

# class RandomGammaTenzor(object):
#     def __init__(self, lower=0.4, upper=2.5):
#         self.lower = lower
#         self.upper = upper
#         assert self.upper >= self.lower, "contrast upper must be >= lower."
#         assert self.lower >= 0, "contrast lower must be non-negative."

#     # expects float image
#     def __call__(self, images, labels=None, mask=None):
#         if random.randint(2):
#             gamma = random.uniform(self.lower, self.upper)
#             images = pow(images, gamma)
#         return images, labels, mask


# class RandomBrightness(object):
#     def __init__(self, delta=26):
#         assert delta >= 0.0
#         assert delta <= 255.0
#         self.delta = delta

#     def __call__(self, images, labels=None, mask=None):
#         if random.randint(2):
#             delta = random.uniform(-self.delta, self.delta)
#             for i in range(0, len(images)):
#                 images[i] += delta
#         return images, labels, mask

# class RandomGaussNoise(object):
#     def __init__(self, std = 5):
#         self.std = std

#     def __call__(self, images, labels=None, mask=None):
#         if random.randint(3) == 0:
#             for i in range(0, len(images)):
#                 mean = ()
#                 std = ()
#                 for k in range(0, images[i].shape[2]):
#                     mean = mean + (0,)
#                     std = std + (self.std,)
#                 noise = np.zeros(images[i].shape, np.float32)
#                 noise = cv2.randn(noise, mean, std)
#                 images[i]+= noise
#         return images, labels, mask

# class Clip(object):
#     def __init__(self, mmin=0., mmax=255.):
#         self.min = mmin
#         self.max = mmax

#     def __call__(self, images, labels=None, mask = None):
#         for i in range(0, len(images)):
#             np.clip(images[i], self.min, self.max, images[i])
#         return images, labels, mask

# class Poisson(object):
#     def __init__(self):
#         self.peak = 0.001

#     def __call__(self, images, labels=None, mask = None):
#         if random.randint(3) == 0:
#             for i in range(0, len(images)):
#                 hsv = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV)
#                 noise =  np.random.poisson(hsv[:,:,0]  * self.peak) / self.peak   # noisy image
#                 # images[i] = images[i] + np.random.poisson(noise)
#                 hsv[:,:,0] = hsv[:,:,0] + noise
#                 images[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#                 # images[i] =  images[i] + noise
#         return images, labels, mask

# class Resize(object):
#     def __init__(self, size=(512, 512)):
#         self.size = size

#     def __call__(self, images, labels=None, mask = None):
#         for i in range(0, len(images)):
#             images[i] = cv2.resize(images[i], self.size, interpolation = cv2.INTER_AREA)
#         if labels is not None:
#             labels = cv2.resize(labels, self.size, interpolation = cv2.INTER_NEAREST)
#         if mask is not None:
#             mask = cv2.resize(mask, self.size, interpolation = cv2.INTER_NEAREST)
#         return images, labels, mask


# class ToCV2Image(object):
#     def __call__(self, tensor, boxes=None, labels=None):
#         return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), labels

# class ToTensorTrain(object):
#     def __call__(self, cvimages, labels=None, mask = None):
#         imgs = [i.astype(np.float32) / 255. for i in cvimages]
#         uncertanty = 0.1
#         if labels is not None:
#             labels = labels.astype(np.float32)

#             new_label = uncertanty * np.random.rand(labels.shape[0], labels.shape[1])
#             label_encoded = np.zeros((2, labels.shape[0],labels.shape[1]), dtype = np.float32)
#             # label_encoded = uncertanty * np.random.rand(2, labels.shape[0],labels.shape[1])
#             label_encoded[0,(labels >= 70) & (labels < 150)] = 1. - new_label[(labels >= 70) & (labels < 150)]
#             label_encoded[0,labels >= 150] = 1.0
#             label_encoded[1,labels >= 150] = 1.0 - new_label[labels >= 150]

#             # label_encoded = np.zeros((2, labels.shape[0],labels.shape[1]), dtype = np.float32)
#             # label_encoded[0,(labels >= 70) & (labels < 150)] = 1.0  -  uncertanty
#             # label_encoded[0,labels >= 150] = 1.0
#             # label_encoded[1,labels >= 150] = 1.0 -  uncertanty

#         # print(uncertanty)
#         # cv2.imshow("m1", label_encoded[0])
#         # cv2.imshow("m2", label_encoded[1])
#         # cv2.waitKey(-1)
#         if mask is not None:
#             mask = mask.astype(np.uint8)
#         if labels is not None and mask is not None:
#             return (torch.from_numpy(np.dstack(imgs)).permute(2, 0, 1), 
#                     torch.from_numpy(label_encoded), 
#                     torch.from_numpy(mask).float())
#         else:
#             return (torch.from_numpy(np.dstack(imgs)).permute(2, 0, 1), 
#                     None, 
#                     None)

# class ToTensor(object):
#     def __call__(self, cvimages, labels=None, mask = None):
#         imgs = [i.astype(np.float32) / 255. for i in cvimages]
#         if labels is not None:
#             # labels = labels.astype(np.float32)
#             # label_encoded = np.zeros((2, labels.shape[0],labels.shape[1]), dtype = np.float32)
#             # label_encoded[0,(labels >= 70) & (labels < 150)] = 1.0
#             # label_encoded[0,labels >= 150] = 1.0
#             # label_encoded[1,labels >= 150] = 1.0
#             labels = labels.astype(np.float32) / 255.

#         # cv2.imshow("m1", label_encoded[0])
#         # cv2.imshow("m2", label_encoded[1])
#         # cv2.waitKey(5)
#         if mask is not None:
#             mask = mask.astype(np.uint8)
#         if labels is not None and mask is not None:
#             return (torch.from_numpy(np.dstack(imgs)).permute(2, 0, 1), 
#                     # torch.from_numpy(label_encoded),
#                     torch.from_numpy(labels), 
#                     torch.from_numpy(mask).float())
#         else:
#             return (torch.from_numpy(np.dstack(imgs)).permute(2, 0, 1), 
#                     None, 
#                     None)

# class ConvertColor(object):
#     def __init__(self, current, transform):
#         self.transform = transform
#         self.current = current

#     def __call__(self, images, labels=None):
#         if self.current == 'BGR' and self.transform == 'HSV':
#             for i in range(0, len(images)):
#                 images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
#         elif self.current == 'RGB' and self.transform == 'HSV':
#             for i in range(0, len(images)):
#                 images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV)
#         elif self.current == 'BGR' and self.transform == 'RGB':
#             for i in range(0, len(images)):
#                 images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
#         elif self.current == 'HSV' and self.transform == 'BGR':
#             for i in range(0, len(images)):
#                 images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2BGR)
#         elif self.current == 'HSV' and self.transform == "RGB":
#             for i in range(0, len(images)):
#                 images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2RGB)
#         else:
#             raise NotImplementedError
#         return images, labels

# class RandomHue(object):
#     def __init__(self, delta=30.0):
#         assert delta >= 0.0 and delta <= 360.0
#         self.delta = delta

#     def __call__(self, images, labels=None, mask = None):
#         if random.randint(2):
#             cvt = ConvertColor(current="RGB", transform='HSV')
#             images, labels = cvt(images, labels)
#             ru = random.uniform(-self.delta, self.delta)
#             for i in range(0, len(images)):
#                 im = images[i]
#                 im[:, :, 0] += ru
#                 im[:, :, 0][im[:, :, 0] > 360.0] -= 360.0
#                 im[:, :, 0][im[:, :, 0] < 0.0] += 360.0
#             cvt = ConvertColor(current="HSV", transform='RGB')
#             images, labels = cvt(images, labels)
#         return images, labels, mask

# class RandomPerspectiveWrap(object):
#     def __init__(self, distortion_scale=0.5, p=0.5, interpolation=3):
#         self.inner_trans = transforms.RandomPerspective(distortion_scale, p, interpolation)

#     def __call__(self, images, labels=None, mask = None):
#         for i in range(0, len(images)):
#             im = Image.fromarray(images[i])
#             im = self.inner_trans(im)
#             images[i] = np.array(im)
#         return images, labels, mask

# class RandomPerspectiveCV(object):
#     def __init__(self, cornerDist = 0.1):
#         self.cornerDist = cornerDist

#     def __call__(self, images, labels=None, mask = None):
#         width, height = images[0].shape[0], images[0].shape[1]
#         rect = np.array([[0,0], 
#                         [width-1,0], 
#                         [width - 1, height - 1], 
#                         [0, height - 1]
#                         ], dtype = "float32")

#         dst = np.array([
#                 [random.randint(self.cornerDist * width), random.randint(self.cornerDist * height)],
#                 [random.randint(width - self.cornerDist * width), random.randint(self.cornerDist * height)],
#                 [random.randint(width - self.cornerDist * width), random.randint(height - self.cornerDist * height)],
#                 [random.randint(self.cornerDist * width), random.randint(height - self.cornerDist * height)]
#                 ], dtype = "float32")
#         M = cv2.getPerspectiveTransform(rect, rect)
#         for i in range(0, len(images)):
#             images[i] = cv2.warpPerspective(images[i], M, (width, height), borderValue = (255, 255, 255))
#         return images, labels, mask

# class Masker(object):
#     def __init__(self, maxDilate = 0.05, alwaysMask = False):
#         self.maxDilate = maxDilate
#         self.alwaysMask = alwaysMask

#     def __call__(self, images, labels=None, mask = None):
#         assert(mask is not None)
#         if random.randint(2) or self.alwaysMask:
#             if mask.shape[0] * self.maxDilate >= 1.0:
#                 dilatation_size = random.randint(int(mask.shape[0] * self.maxDilate))
#                 element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
#                 mask = cv2.dilate(mask, element)
#             msk = mask > 10

#             for i in range(0, len(images)):
#                 im = images[i]
#                 im[msk] = 255
#             labels[msk] = 0
#         return images, labels, mask

# def odd(f):
#     return int(np.ceil(f) // 2 * 2 + 1)

# def makeBlob(img_shape, max_blob_size, offset, randomize_size = True, randomize_pos = True):

#     if randomize_size:
#         scale = random.randint(img_shape[0] / 7, max_blob_size / 2)
#     else:
#         scale = max_blob_size / 2

#     if randomize_pos:
#         blob_pos = (random.randint(offset + scale, img_shape[1] - offset - scale), 
#                     random.randint(offset + scale, img_shape[0] - offset - scale))
#     else:
#         blob_pos = (img_shape[1] / 2, img_shape[0] / 2)
    
#     N = 6
#     amps = random.sample(N) * (0.75 / N)
#     phases = random.sample(N) * 2 * np.pi
#     n_seq = np.arange(0, N, 1) + 1
     
#     angles = np.arange(0, 2 * np.pi, np.radians(1), dtype = np.float)
     
#     radiuses = [(np.sum(amps * np.cos(n_seq * a + phases)) + 1) * scale for a in angles]
#     radiuses = np.array(radiuses, dtype = np.float) 
     
#     pts = [ [np.cos(a)* r+blob_pos[0], np.sin(a)* r+blob_pos[1]] for a,r in zip(angles, radiuses)]
#     pts = np.asarray(pts).astype(np.int32)
    
#     image = np.full(img_shape, 255, dtype = np.uint8)
#     cv2.fillPoly(image, [pts], 0)
#     return image

# class alterNonDirtMask(object):
#     def __init__(self,  blobSize = 0.7):
#         self.blobSize = blobSize

#     def __call__(self, images, labels=None, mask = None):
#         if random.randint(2):
#             return images, labels, mask

#         blb_size = 1.2 #random.uniform(self.blobSize, 1)
#         height, width, __ = images[0].shape

#         blob = makeBlob(mask.shape,
#                            height * blb_size, 
#                            0, 
#                            randomize_size = False, 
#                            randomize_pos = False)
      
#         mask = np.maximum(mask, blob)
        
#         return images, labels, mask

# class Rotate(object):
#     def __call__(self, images, labels=None, mask = None):
#         angle = random.uniform(-15, 15)
#         image_center = tuple(np.array(images[0].shape[1::-1]) / 2)
#         rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#         for i in range(0, len(images)):   
#             images[i] = cv2.warpAffine(images[i], rot_mat, images[i].shape[1::-1], flags=cv2.INTER_CUBIC, 
#                 borderMode=cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))
#             # images[i] = rotateAndScale(images[i], scaleFactor = 1, degreesCCW = angle)
#         mask = cv2.warpAffine(mask, rot_mat, images[i].shape[1::-1], flags=cv2.INTER_NEAREST, borderValue = (255))
# #         labels = cv2.warpAffine(labels, rot_mat, labels.shape[1::-1], flags=cv2.INTER_CUBIC, borderValue = (0))
#         labels = cv2.warpAffine(labels, rot_mat, labels.shape[1::-1], flags=cv2.INTER_NEAREST, borderValue = (0))
#         return images, labels, mask
