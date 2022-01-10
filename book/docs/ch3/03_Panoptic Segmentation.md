# Overview

## Introduction

## Performance Measure

$$
\text{PQ}=\frac{\sum_{(p,q) \in \text{TP}}\text{IoU}(p,q)}{|\text{TP}|+\frac{1}{2}|\text{FP}|+\frac{1}{2}|\text{FN}|}=\frac{\sum_{(p,q) \in \text{TP}}\text{IoU}(p,q)}{|\text{TP}|}\times\frac{\text{TP}}{{|\text{TP}|+\frac{1}{2}|\text{FP}|+\frac{1}{2}|\text{FN}|}}
$$

## Trend

:::{figure-md} performance-pano
<img src="pic/panoSeg1.png" alt="panoSeg1" class="bg-primary mb-1" width="800px">

Evolution of Panoptic Segmentation
:::

*Latest update: Jan 6, 2022*