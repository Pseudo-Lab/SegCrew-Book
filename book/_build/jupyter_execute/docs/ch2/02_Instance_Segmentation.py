#!/usr/bin/env python
# coding: utf-8

# # Overview
# 

# ## Introduction
# 
# :::{figure-md} markdown-fig
# <img src="pic/instSeg2.png" alt="pfpn1" class="bg-primary mb-1" width="600px">
# 
# Semantic Segmentation vs. Instance Segmentation (source: ??)
# :::
# 
# - Instance segmentation은 입력 영상에 존재하는 객체(instance)의 정보를 반영한 pixel level classification 방법으로, 각 pixel이  속한 객체 label 및 객체의 classes를 추정하는 방법이다. 
# - semantic segmentation과 같이 pixel-level의 classification으로 정의되는 것은 동일하지만, 같은 class에 속하더라도 객체별로 다른 label을 가져야 하기 때문에 위치 정보가 반영된 inference가 추가적으로 필요하다.
# - 따라서 object detection을 통해 얻은 bounding box 내의 segmentation을 수행하는 방법이 주로 적용되어, 일반적으로 instance segmentation은 배경을 제외한 객체(instance)의 영역에 대해서 class와 label을 예측한다.

# ## Performance Measure
# 
# - Instance segmentation 성능평가를 위해 주로 적용되는 measure는 COCO banchmark의 평가지표로 사용되는 $\text{mask}\ AP$(average precision)으로 instance의 ground truth mask와 prediction된 mask의 IoU threshold를 0.50에서 0.05단위로 0.95까지 설정하여 구한 precision의 평균으로 구한다. COCO style의 mAP(mean AP)는 다음과 같이 계산할 수 있다. 
# 
#     $$mAP=\frac{1}{C} \sum_{c \in C}\frac{|TP_c|}{|FP_c|+|TP_c|}$$
# 
#     여기서 $TP$는 true positive, $FP$는 false positive를 나타내며, $C$는 ground truth의 class 수이다. 
#     
# - IoU threshold가 0.5 (50%)인 값을 $AP_{50}$으로, threshold가 0.7 (70%) 일 때는 $AP_{70}$으로 표현한다. 
# 

# ## Trend

# In[11]:


#@title
# https://medium.com/analytics-vidhya/creating-a-dual-axis-pareto-chart-in-altair-e3673107dd14
# https://altair-viz.github.io/user_guide/interactions.html
# https://www.datacamp.com/tutorial/altair-in-python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import altair as alt
import pandas as pd


def GetGraphElement(chart_title, data, x_scale, y_scale, perf_measure, line_color = "#000000", point_color = "#000000", text_color = "#000000", text_y_pos = -10, textperf_y_pos=-20):
    base = alt.Chart(data).encode(
    x = alt.X("idx", scale=alt.Scale(domain=x_scale),axis=None),
    ).properties (
    width = 800,
    title = [chart_title]
    )

    line = base.mark_line(strokeWidth= 1.5, color = line_color).encode(
        y=alt.Y(perf_measure, scale=alt.Scale(domain=y_scale),axis=alt.Axis(grid=True)),
        color=alt.Color('type'),
    )

    points = base.mark_circle(strokeWidth= 3, color = point_color).encode(
            y=alt.Y(perf_measure, scale=alt.Scale(domain=y_scale), axis=None),
            tooltip = [alt.Tooltip('year'),
            alt.Tooltip('nickname'),
            alt.Tooltip(perf_measure),
            alt.Tooltip('note'),],
    )

    point_nick = points.mark_text(align='center', baseline='middle', dy = text_y_pos,).encode(
        y= alt.Y(perf_measure, scale=alt.Scale(domain=y_scale), axis=None),
        text=alt.Text(perf_measure),
        color= alt.value(text_color)
    )
    point_perf = points.mark_text(align='center', baseline='middle', dy = textperf_y_pos).encode(
        y= alt.Y(perf_measure, scale=alt.Scale(domain=y_scale), axis=None),
        text=alt.Text('nickname'),
        color= alt.value(text_color)
    )   

    return base, line, points, point_nick, point_perf

def description_test(pos_x, pos_y, text, color):
    return alt.Chart({'values':[{}]}).mark_text(align = "left", baseline ="top").encode(
        x = alt.value(pos_x),
        y = alt.value(pos_y),
        text = alt.value([text]),
        color= alt.value(color)
    )
    


# In[15]:


data = pd.read_csv("instance_trend_cocotest.csv", sep=",")

anchor_data = data.loc[data['type'] =="Anchor/proposal based method"]
single_data = data.loc[data['type'] =="Single-stage/Anchor-free"]
trans_data = data.loc[data['type'] =="Transformer"]

perf_measure = 'maskAP'

x_scale = [0,data['idx'].max()+1]
y_scale = [((data[perf_measure].min()//5))*5,((data[perf_measure].max()//5)+1)*5]

chart_title = "Trend on AP (COCO Test-dev)"

base, line, points, point_nick, point_perf = GetGraphElement(chart_title, anchor_data, x_scale, y_scale, perf_measure, 
                                                            line_color = "#fde725", point_color = "#000000", text_color = "#000000", 
                                                            text_y_pos = -20, textperf_y_pos=-30)
base2, line2, points2, point_nick2, point_perf2 = GetGraphElement(chart_title, single_data, x_scale, y_scale, perf_measure, 
                                                                    line_color = "#cb4154", point_color = "#000000", text_color = "#000000", 
                                                                    text_y_pos = -10, textperf_y_pos=20)
base3, line3, points3, point_nick3, point_perf3 = GetGraphElement(chart_title, trans_data, x_scale, y_scale, perf_measure, 
                                                                    line_color = "#3b518b", point_color = "#000000", text_color = "#000000", 
                                                                    text_y_pos = 20, textperf_y_pos=30)
(
    line+points+point_nick+point_perf+
    line2+points2+point_nick2+point_perf2+ 
    line3+points3+point_nick3+point_perf3
).resolve_scale(y = 'independent')


# ## Reference
# - https://www.cityscapes-dataset.com/benchmarks/
# - https://sviro.kl.dfki.de/instance-segmentation/
# - https://pyimagesearch.com/2022/05/02/mean-average-precision-map-using-the-coco-evaluator/
# - https://sviro.kl.dfki.de/instance-segmentation/
# 
# 
# *Latest update: Nov. 9, 2022*
