# Overview

## Introduction

:::{figure-md} pano-fig
<img src="pic/panoSeg3.PNG" alt="pfpn2" class="bg-primary mb-1" width="600px">

Segmentation Tasks (source: [arXiv:2006.12567](https://arxiv.org/abs/2006.12567))
:::

- Panoptic Segmentation은 pixel level의 classification을 수행하는 semantic segmentation과 객체 단위 기반 classfication을 수행하는 instance segmentation을 통합한 task로 각 pixel을 배경에 해당하는 stuff와 객체(instance)에 해당하는 things class로 분류하는 task이다. 
- 즉 입력 영상의 각 pixel을 overlap되지 않은 class label로 분류하는 문제로 정의할 수 있다. 

## Performance Measure

- Panoptic Segmentation의 성능을 평가히기 위해 PQ(Panoptic Quality)를 사용한다. 

$$
\text{PQ}=\frac{\sum_{(p,q) \in \text{TP}}\text{IoU}(p,q)}{|\text{TP}|+\frac{1}{2}|\text{FP}|+\frac{1}{2}|\text{FN}|}
$$

- PQ를 계산하기 위해 먼저 Segment matching을 수행한 후 matching된 segment에 대해 PQ를 계산한다. 이때 GT와 predicted segment와의 match 여부는 IoU가 0.5 이상이고 가장 큰 IoU를 가지는 segment를 유일한 matched segment로 판정한다. 
- Metched segment를 계산한 후 TP(true positivie), FP(false positive)와 FN(false negative)를 구한다. 이는 {numref}`gt-fig`와 같이 나타낼 수 있다

:::{figure-md} gt-fig
<img src="pic/panoSeg2.png" alt="pfpn2" class="bg-primary mb-1" width="500px">

GT and predicted panoptic segmentations of an image (source: arXiv:1801.00868)
:::

- PQ는 match되는 segment에 대한 평균 IoU에 match되지 않는 segment에 대한 페널티 $\left(\frac{1}{2}|FP| +\frac{1}{2} |FN |\right)$ 가 추가된 형태로 구성되어 있다.  PQ를 TP 항을 추가하여 분리하면 SQ(segmentation Quaility) 항과 RQ(recognition quality) 항의 곱으로 표현 가능하다. RQ는 널리 사용되는 F1 score로 해석할 수 있다. 

$$
\text{PQ}=\underbrace{\frac{\sum_{(p,q) \in \text{TP}}\text{IoU}(p,q)}{|\text{TP}|}}_{\text{SQ}}\times \underbrace{\frac{\text{TP}}{{|\text{TP}|+\frac{1}{2}|\text{FP}|+\frac{1}{2}|\text{FN}|}}}_{\text{RQ}}
$$

- PQ 계산시 void labels (unknown pixels 또는 out of class pixel)은 계산과정에서 제외한다.

## Trend

:::{figure-md} performance-pano
<img src="pic/panoSeg1.png" alt="panoSeg1" class="bg-primary mb-1" width="800px">

Evolution of Panoptic Segmentation
:::

- {numref}`Fig. %s <performance-pano>`는 COCO Test-Dev set으로 평가한 panoptic segmentation 성능이다.

+++

- 본 chapter에서는 panoptic segmentation 논문을 i) box based methods와 ii) box-free methods로 구분하여 리뷰하고 각 지금까지의 발전 현황과 개선방안에 대해 고찰한다.

*Latest update: Jun 7, 2022*