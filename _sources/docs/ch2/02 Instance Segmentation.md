# Overview

Draft version

## Introduction

:::{figure-md} markdown-fig
<img src="pic/instSeg2.png" alt="pfpn1" class="bg-primary mb-1" width="600px">

Semantic Segmentation vs. Instance Segmentation (source: ??)
:::

- Instance segmentation은 입력 영상에 존재하는 객체(instance)의 정보를 반영한 pixel level classification 방법으로, 각 pixel이  속한 객체 label 및 객체의 classes를 추정하는 방법이다. 
- semantic segmentation과 같이 pixel-level의 classification으로 정의되는 것은 동일하지만, 같은 class에 속하더라도 객체별로 다른 label을 가져야 하기 때문에 위치 정보가 반영된 inference가 추가적으로 필요하다.
- 따라서 object detection을 통해 얻은 bounding box 내의 segmentation을 수행하는 방법이 주로 적용되어, 일반적으로 instance segmentation은 배경을 제외한 객체(instance)의 영역에 대해서 class와 label을 예측한다.

## Performance Measure

- Instance segmentation 성능평가를 위해 주로 적용되는 measure는 COCO banchmark의 평가지표로 사용되는 $\text{mask}\ AP$(average precision)으로 instance의 ground truth mask와 prediction된 mask의 IoU threshold를 0.50에서 0.05단위로 0.95까지 설정하여 구한 precision의 평균으로 구한다. COCO style의 mAP(mean AP)는 다음과 같이 계산할 수 있다. 
    $$mAP=\frac{1}{C} \sum_{c \in C}\frac{|TP_c|}{|FP_c|+|TP_c|}$$
    여기서 $TP$는 true positive, $FP$는 false positive를 나타내며, $C$는 ground truth의 class 수이다. 
- IoU threshold가 0.5 (50%)인 값을 $AP_{50}$으로, threshold가 0.7 (70%) 일 때는 $AP_{70}$으로 표현한다. 

## Trend

:::{figure-md} performance-instseg
<img src="pic/instSeg1.png" alt="instSeg1" class="bg-primary mb-1" width="600px">

Examples of AP?? (source: arXiv:1407.1808)
:::
맞는지 확인할 것...

## Reference
- https://www.cityscapes-dataset.com/benchmarks/
- https://sviro.kl.dfki.de/instance-segmentation/
- https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/
- https://sviro.kl.dfki.de/instance-segmentation/

*Latest update: July 13, 2022*