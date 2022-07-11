"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch import optim
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import numpy as np
import cv2
import sys

from models.model_depth import *
from models.refinement_net import *
from models.modules import *
from datasets.plane_semantic_dataset import *

from utils import *
from visualize_utils import *
from evaluate_utils import *
from options import parse_args
from config import PlaneConfig

import logging
import time

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"

def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join("logs",
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter) 
    logger.addHandler(fHandler) 

    return logger


    
def train(options):
    logger = getLogger()
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    local_rank = options.local_rank

    config = PlaneConfig(options)
    
    # Set DDP variables
    if local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl')

    dataset = PlaneDatasetSemantic(options, config, split='train', random=True)

    print('the number of images', len(dataset))

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if local_rank != -1 else None
    dataloader = DataLoader(dataset, batch_size=1, shuffle=(True & local_rank == -1), num_workers=16, sampler=train_sampler)

    model = MaskRCNN(config)
    model.cuda()
    model.train()    

    if options.restore == 1:
        ## Resume training
        print('restore')
        #model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))

    elif options.restore == 2:
        ## Train upon Mask R-CNN weights
        if local_rank in [-1, 0]:
            model_path = "/data/wudong/raw_project/planercnn/checkpoint/planercnn_normal_warping_refine/checkpoint.pth"
            print("Loading pretrained weights ", model_path)
            model.load_weights(model_path)
        pass
    
    if options.trainingMode != '':
        ## Specify which layers to train, default is "all"
        if local_rank in [-1, 0]:
            layer_regex = {
                ## all layers but the backbone
                "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                ## From a specific Resnet stage and up
                "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                ## All layers
                "all": ".*",
                "classifier": "(classifier.*)|(mask.*)|(depth.*)",
            }
            assert(options.trainingMode in layer_regex.keys())
            layers = layer_regex[options.trainingMode]
            model.set_trainable(layers)
        pass

    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]

    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=options.LR, momentum=0.9)
    
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth'))        
        pass

    if local_rank != -1:
        group_model = torch.distributed.new_group(ranks=[0,1,2,3,4,5])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True,process_group=group_model,broadcast_buffers=True)
        
    for epoch in range(options.numEpochs):
        if local_rank != -1:
            dataloader.sampler.set_epoch(epoch)

        epoch_losses = []
        data_iterator = tqdm(dataloader, total=len(dataset) + 1)

        optimizer.zero_grad()

        for sampleIndex, sample in enumerate(data_iterator):
            losses = []            

            input_pair = []
            detection_pair = []
            dicts_pair = []

            camera = sample[32][0].cuda()                
            for indexOffset in [0, 14]:
                images, image_metas, rpn_match, rpn_bbox, gt_semantic_ids, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, gt_plane, gt_segmentation, plane_indices = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda(), sample[indexOffset + 12].cuda(), sample[indexOffset + 13].cuda()
                
                droid_depth = torch.randn([1,1,640,640] , dtype=images.dtype).to(images.device)

                if indexOffset == 14:
                    input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'plane': gt_plane, 'camera': camera})
                    continue
                #results =model.module.predict([images, image_metas, gt_semantic_ids, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='training_detection', use_nms=2, use_refinement='refinement' in options.suffix, return_feature_map=True)
                results =model.predict([[images, droid_depth*128], image_metas, gt_semantic_ids, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='training_detection', use_nms=2, use_refinement='refinement' in options.suffix, return_feature_map=True)
                rpn_class_logits, rpn_pred_bbox, target_semantic_ids, mrcnn_semantic_logits, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, feature_map, depth_np_pred = results
                rpn_class_loss, rpn_bbox_loss, mrcnn_semantic_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, mrcnn_parameter_loss = compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_semantic_ids, mrcnn_semantic_logits, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters)

                losses += [rpn_class_loss , rpn_bbox_loss ,  mrcnn_semantic_loss , mrcnn_class_loss , mrcnn_bbox_loss , mrcnn_mask_loss , mrcnn_parameter_loss]

                # config.PREDICT_NORMAL_NP : False
                if config.PREDICT_NORMAL_NP:
                    normal_np_pred = depth_np_pred[0, 1:]                    
                    depth_np_pred = depth_np_pred[:, 0]
                    gt_normal = gt_depth[0, 1:]                    
                    gt_depth = gt_depth[:, 0]
                    depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                    normal_np_loss = l2LossMask(normal_np_pred[:, 80:560], gt_normal[:, 80:560], (torch.norm(gt_normal[:, 80:560], dim=0) > 1e-4).float())
                    losses.append(depth_np_loss)
                    losses.append(normal_np_loss)
                else:
                    depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                    losses.append(depth_np_loss)
                    normal_np_pred = None
                    pass
                
                # len(detections) > 0   True
                if len(detections) > 0:
                    detections, detection_masks = unmoldDetections(config, camera, detections, detection_masks, depth_np_pred, normal_np_pred, debug=False)
                    #print(detections.shape)         #torch.Size([1, 11])
                    #print(detection_masks.shape)    #torch.Size([1, 640, 640])
                    XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
                    """
                    print(XYZ_pred.shape)           #torch.Size([3, 640, 640])
                    print(detection_mask.shape)     #torch.Size([640, 640])
                    print(plane_XYZ.shape)          #torch.Size([22, 3, 640, 640])
                    """
                    detection_mask = detection_mask.unsqueeze(0)                        
                else:
                    XYZ_pred = torch.zeros((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                    detection_mask = torch.zeros((1, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                    plane_XYZ = torch.zeros((1, 3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()                        
                    pass

                """
                print(XYZ_pred.shape)           torch.Size([3, 640, 640])
                print(detection_mask.shape)     torch.Size([1, 640, 640])
                print(plane_XYZ.shape)          torch.Size([16, 3, 640, 640])
                """
                input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'parameters': detection_gt_parameters, 'plane': gt_plane, 'camera': camera})
                detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'feature_map': feature_map[0], 'plane_XYZ': plane_XYZ, 'depth_np': depth_np_pred})

            ## The warping loss
            for c in range(1, 2):
                if 'warping' not in options.suffix:
                    break

                detection_dict = detection_pair[1 - c]
                neighbor_info = torch.cat([detection_dict['XYZ'], detection_dict['mask'], input_pair[1 - c]['image'][0]], dim=0).unsqueeze(0)
                warped_info, valid_mask = warpModuleDepth(config, camera, input_pair[c]['depth'][0], neighbor_info, input_pair[c]['extrinsics'][0], input_pair[1 - c]['extrinsics'][0], width=config.IMAGE_MAX_DIM, height=config.IMAGE_MIN_DIM)

                XYZ = warped_info[:3].view((3, -1))
                XYZ = torch.cat([XYZ, torch.ones((1, int(XYZ.shape[1]))).cuda()], dim=0)
                transformed_XYZ = torch.matmul(input_pair[c]['extrinsics'][0], torch.matmul(input_pair[1 - c]['extrinsics'][0].inverse(), XYZ))
                transformed_XYZ = transformed_XYZ[:3].view(detection_dict['XYZ'].shape)
                warped_depth = transformed_XYZ[1:2]
                warped_images = warped_info[4:7].unsqueeze(0)
                warped_mask = warped_info[3]

                with torch.no_grad():                    
                    valid_mask = valid_mask * (input_pair[c]['depth'] > 1e-4).float()
                    pass

                warped_depth_loss = l1LossMask(warped_depth, input_pair[c]['depth'], valid_mask)
                losses += [warped_depth_loss]
                
                input_pair[c]['warped_depth'] = (warped_depth * valid_mask + (1 - valid_mask) * 10).squeeze()
                continue            
            loss = sum(losses)
            losses = [l.data.item() for l in losses]
            
            epoch_losses.append(losses)
            status = str(epoch + 1) + ' loss: '
            for l in losses:
                status += '%0.5f '%l
                continue

            status += '%0.5f '%loss.data.item()

            #sys.stdout.write('\r ' + str(sampleIndex) + ' ' + status)
            #sys.stdout.flush()

            if sampleIndex % 50 == 0:
                logger.info(str(sampleIndex) + ' ' + status)

            data_iterator.set_description(status)

            loss.backward()
            
            if (sampleIndex + 1) % options.batchSize == 0:
                optimizer.step()
                optimizer.zero_grad()
                pass

            if sampleIndex % 500 < options.batchSize or options.visualizeMode == 'debug':
                ## Visualize intermediate results
                visualizeBatchPair(options, config, input_pair, detection_pair, indexOffset=sampleIndex % 500)
                if options.visualizeMode == 'debug' and sampleIndex % 500 >= options.batchSize - 1:
                    exit(1)
                    pass
                pass

            if (sampleIndex + 1) % 1000 == 0:
                ## Save models
                logger.info(str(sampleIndex)+" average loss: " + str(np.array(epoch_losses).mean(0)))
                epoch_losses = []
                torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
                torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')
                pass
            continue
        continue
    return


if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'exp2'

    args.keyname += '_' + args.anchorType
    if args.dataset != '':
        args.keyname += '_' + args.dataset
        pass
    if args.trainingMode != 'all':
        args.keyname += '_' + args.trainingMode
        pass
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    if False:
        writeHTML(args.test_dir, ['image_0', 'segmentation_0', 'depth_0', 'depth_0_detection', 'depth_0_detection_ori'], labels=['input', 'segmentation', 'gt', 'before', 'after'], numImages=20, image_width=160, convertToImage=True)
        exit(1)
        
    os.system('rm ' + args.test_dir + '/*.png')
    print('keyname=%s task=%s started'%(args.keyname, args.task))

    train(args)

