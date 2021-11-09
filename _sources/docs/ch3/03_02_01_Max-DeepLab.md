# [Draft] MaX-DeepLab - CVPR 21

Title: End-to-End Panoptic Segmentation with Mask Transformers

Review By: Jeunghyun Byun

Edited by: 

## Reference

- Paper: [https://arxiv.org/abs/2012.00759](https://arxiv.org/abs/2012.00759)
- Code: [https://github.com/google-research/deeplab2](https://github.com/google-research/deeplab2)
- review: [https://ai.googleblog.com/2021/04/max-deeplab-dual-path-transformers-for.html](https://ai.googleblog.com/2021/04/max-deeplab-dual-path-transformers-for.html),

# Problem Statement

Panoptic segmentation task이란:

- "overlap하지 않는 각각의 class label에 해당되는 mask를 구하는 것"

**Panoptic** segmentation = Merging **semantic (assigning a class label to each pixel) + instance** (**detecting and segmenting each object instance**) segmentation results

# Contribution

- MaX-DeepLab is the **first end-to-end model for panoptic segmentation (초록색으로 색칠된 부분 참고)**:
    - 기존의 방법들은 surrogate sub-task들 (e.g. box detection, anchor design rules, non-maximum suppression, 등등)을 포함한 pipeline을 사용함 (즉, hand-coded 된 prior에 의존)
- MaX-DeepLab은 mask transformer을 사용하고 bipartite matching으로부터 inspire된 loss을 사용함
- Dual-path transformer 사용
- COCO dataset에 대해서 새로운 SOTA 결과

# Background

(Transformer에 대한 설명은 생략! 이준형님께서 잘 정리해주셔서 [여기](https://www.notion.so/9398a7eabb1048a1ab98993e02a9fd1d) 참고하면 될 것 같습니다! )

기존의 Panoptic Segmentation method:

- 여러 surrogate sub-task들에 의존하여 panoptic segmentation mask 결과를 구함.
- 즉 기존의 panoptic segmentation은 한번에 결과값을 구하는 것이 아니라 여러 surrogate-task들로 구성되어 있는 다소 복잡한 pipeline **(빨간색으로 색칠된 부분)** 으로 구성되어 있다.
- e.g. anchors, box assignment rules, non-maximum suppression, thing-stuff merging
    
    ![maxdeeplab1.png](pic/max-deeplab/maxdeeplab1.png)
    

복잡한 pipeline의 문제점:

- 각각의 surrogate task들로부터 "undesired artifact"가 발생할 수 있고 pipeline을 거치면서 더 큰 artifact / noise 로 번질 수 있다. **(빨간색으로 색칠된 부분)**

(Box based pipeline, box free methods에 대해서 살펴볼 필요 있음!)

최근의 여러 연구들이 **pipeline을 간단화하는데** 많이 노력함

## Box Based Method in Panoptic Segmentation

먼저 **Box based method가** 있다. 이 방법은 Mask R-CNN과 같은 object detection 모델로 먼저 object bounding boxes들을 구하고 각 box안의 object에 해당되는 mask을 구한다. **Instance segmentation**와 **Semantic segmentation**가 서로 merge돼서 최종적인 panoptic segmentation을 구하게 된다.

Exemplary Works:

- **Panoptic FPN**: 대표적인 모델
- **UPSNet**: parameter-free panoptic head
- **DETR**: end-to-end approach for box detection (to replace detectors in panoptic segmentation
    - **한계**: **training process still relies heavily on box detection**
- **DetectoRS**: SOTA work in Box Based methods. Uses recursive feature pyramid and switachable atrous convolution.

## Box Free Method in Panoptic Segmentation

그리고 **Box free method**들이 있다. 먼저 semantic segment들을 구하고, 각각의 semantic segment에 속해 있는 pixel들을 group하여 instance segment들을 구한다. Grouping 방법으로는 예를 들어:

- Instance Center Regression
- Watershed transform
- Hough-voting
- Pixel affinity가 있다.

 Exemplary Works:

- **Axial-DeepLab**: predict pixel-wise offsets to pre-defined instance centers **(Panoptic-DeepLab + Fully axial-attention)**
    - **한계: challenges with highly deformable objects (which have a large variety of shapes) or nearby objects with close centers in the image**
- 비슷한 맥락의 논문으로: Deeperlab Single-shot image parser, Leveraging Semi-Supervised Learning in Video Sequneces for Urban Scene Segmentation (ECCV 2020), Semantic image-segmentation with deep convolutional nets and fully connected crfs (ICLR 2015) 등등이 있다.

Box based method. Box free method들 모두 일종의 "hand-coded prior"을 사용하므로 end-to-end pipeline이 아니다.

# Proposed Method

## Notations:

Class-labeled masks:

![maxdeeplab2.png](pic/max-deeplab/maxdeeplab2.png)

$\hat{m}_i$ = predicted masks

$\hat{p}_i$ = predicted probability of assigning class c to mask hat(m)_i

## PQ-style loss

![maxdeeplab3.png](pic/max-deeplab/maxdeeplab3.png)

PQ = panoptic quality, RQ = recognition quality, SQ = segmentation quality

![maxdeeplab4.png](pic/max-deeplab/maxdeeplab4.png)

(Dice coefficient definition)

## Mask Matching

To assign a predicted mask to each ground truth, solve a one-to-one bipartite matching problem between the prediction set and the ground truth set. Use **Hungarian algorithm**.

![maxdeeplab5.png](pic/max-deeplab/maxdeeplab5.png)

- K matches predictions = positive masks
- (N-K) masks left = negative masks (i.e. no object)

## Loss

For positive masks, we need to maximize (substitue mask matching equation's similarity with equation in PQ-style loss)

![maxdeeplab6.png](pic/max-deeplab/maxdeeplab6.png)

In practice, apply product rule of gradient and change probability into log p (since log p aligns with the common cross-entropy loss that scales gradients better for optimization):

![maxdeeplab7.png](pic/max-deeplab/maxdeeplab7.png)

Loss for negative masks Total Loss:

![maxdeeplab8.png](pic/max-deeplab/maxdeeplab8.png)

![maxdeeplab9.png](pic/max-deeplab/maxdeeplab9.png)

Total Loss: 

![maxdeeplab10.png](pic/max-deeplab/maxdeeplab10.png)

## Model Architecture

![maxdeeplab11.png](pic/max-deeplab/maxdeeplab11.png)

**components:** dual-path transformer, stacked decoder, output heads (for mask and classes prediction)

![maxdeeplab12.png](pic/max-deeplab/maxdeeplab12.png)

**Dual-path**: Augment a **2D pixel-based CNN (H x W x d_in)** with a **1D global memory (N x d_in)** of size N (i.e. total number of predictions)

Types of communications:

- traditional memory-to-pixel (M2P) attention
- memory-to-memory (M2M) self-attention
- pixel-to-memory (P2M) feedback attention
- pixel-to-pixel (P2P) self-attention (axial attention blocks)

Feedback attention's output **(Pixel-to-memory)** at pixel position a:

![maxdeeplab13.png](pic/max-deeplab/maxdeeplab13.png)

Output of **memory-to-pixel (M2P)** and **memory-to-memory (M2M)** attention:

![maxdeeplab14.png](pic/max-deeplab/maxdeeplab14.png)

**Stacked Decoder:** stack L times traversing through (4, 8 and 16 output strides)

- Note: different to feature pyramid networks, the stacked decoder in this model does not directly use intermediate pyramidal features for prediction!

**Output heads:**

- Memory Path:
    - predict mask classes with 2 FC and softmax (N x C) → **p(c)**
    - another 2FCs predict mask features → **feature f** (N x D)
- Pixel Path:
    - employ 2 Convs to produce normalized → **feature g** ( D x H/4 x W/4)

Then, mask prediction m:

![maxdeeplab15.png](pic/max-deeplab/maxdeeplab15.png)

Mask prediction inspired by CondInst and SOLOv2

## Auxiliary Losses

- Instance discrimination: helps cluster decoder features into instances
- Mask-ID Cross-Entropy: classifies each pixel in N masks
- Semantic Segmentation

# Experimentation Result

Computing resources:

- 32 TPU cores for 100k iterations (54 epochs)

Data: COCO

Models: 

- **Max-DeepLab-L** (Wide-ResNet-41, L=2 stacking)
- **Max-DeepLab-S** (ResNet-50 with axial attention blocks, no stacking L=0)
    
    ![maxdeeplab16.png](pic/max-deeplab/maxdeeplab16.png)

    ![maxdeeplab17.png](pic/max-deeplab/maxdeeplab17.png)

    ![maxdeeplab18.png](pic/max-deeplab/maxdeeplab18.png)

