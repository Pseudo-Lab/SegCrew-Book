# Overview

## Introduction

## Performance Measure

- Panoptic Segmentation의 성능을 평가히기 위해 Semantic/Instance Segmentation에서 적용되는 measure를 동시에 적용하는 것은 communication의 혼란과 알고리즘간의 비교를 어렵게 하기 때문에 통일된 지표로 제안된 PQ(Panoptic Quality)를 이용하여 평가한다. 

$$
\text{PQ}=\frac{\sum_{(p,q) \in \text{TP}}\text{IoU}(p,q)}{|\text{TP}|+\frac{1}{2}|\text{FP}|+\frac{1}{2}|\text{FN}|}
$$

- 여기서 TP(true positivie), FP(false positive)와 FN(false negative)는 {numref}`gt-fig`와 같이 나타낼 수 있다.

:::{figure-md} gt-fig
<img src="pic/panoSeg2.png" alt="pfpn2" class="bg-primary mb-1" width="500px">

GT and predicted panoptic segmentations of an image (source: arXiv:1801.00868)
:::

- PQ는 match되는 segment에 대한 평균 IoU에 match되지 않는 segment에 대한 페널티 $\left(\frac{1}{2}|FP| +\frac{1}{2} |FN |\right)$  가 추가된 형태로 구성된다. 
- 또한 PQ를 TP 항을 추가하여 분리하면 SQ(segmentation Quaility) 항과 RQ(recognition quality) 항의 곱으로 표현 가능하다. RQ는 널리 사용되는 F1 score이다.

$$
\text{PQ}=\frac{\sum_{(p,q) \in \text{TP}}\text{IoU}(p,q)}{|\text{TP}|}\times\frac{\text{TP}}{{|\text{TP}|+\frac{1}{2}|\text{FP}|+\frac{1}{2}|\text{FN}|}}
$$

## Trend

:::{figure-md} performance-pano
<img src="pic/panoSeg1.png" alt="panoSeg1" class="bg-primary mb-1" width="800px">

Evolution of Panoptic Segmentation
:::

*Latest update: Jan 6, 2022*