#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# ## Introduction
# 
# :::{figure-md} markdown-fig
# <img src="pic/weakly1.png" alt="weakly1" class="bg-primary mb-1" width="800px">
# 
# Weakly supervised learning. (source: Hakan, CVPR18 Tutorial)
# :::
# 
# - Weakly-supervised learning은 full supervision이 아닌 완벽하지 않은 저수준(lower degree, noisy, limited, or imprecise sources)의 supervision 이용하여 학습방법이다.
# - Segmentation task의 경우 supervised leaning 기반 학습 setting에서는 학습데이터로 주어진 모든 영상 정보에 대해 pixel level의 class label이 필요하며, 이러한 고수준의 labeled data가 필요한 task에서 data를 확보하는 과정에서 다음과 같은 문제가 발생할 수 있다.
#     
#     (1) 충분한 양의 labeled data를 확보하기 어렵다.
# 
#     (2) 특정 task의 경우 label data 생성에 전문가가 필요하고, 이는 많은 비용과 노력을 필요로 한다. (e.g.의료 영상)
# 
#     (3) 충분한 Data를 수집하기 위한 시간이 많이 든다.
#     
# - Weakly supervised learning 기반 방법에서는 영상 전체의 classification 결과 (class label), 또는 object detection 결과 (bounding box, class label)와 같은 full supervision 대비 저수준의 labeled data를 이용하여 semantic/instance segmentation task와 같은 고수준의 task를 수행하는 network를 학습한다. 이를 통해 data 수집에 필요한 비용과 시간을 절감할 수 있다.
# - Pixel/instance level의 label이 확보되지 않는 weakly supervised setting에서는 Class attention map (CAM)을 주로 활용한다.
# 
# :::{figure-md} markdown-fig
# <img src="pic/weakly2.png" alt="weakly2" class="bg-primary mb-1" width="800px">
# 
# Class Attention Map (source: Learning Deep Features for Discriminative Localization, CVPR 2016)
# :::
# 
# - CAM은 Network가 image를 특정 class로 prediction하게 하는 feature 내의 위치 정보를 표현한다.
# - 마지막 conv. layer의 output feature map $f^{\text{CAM}}$ 특정 class로 분류될 activation value를 구하기 위한 weight와 weighted sum을 수행하여 CAM을 도출한다.
# 
# - 이러한 CAM 정보는 class label만이 주어진 환경에서 이미지 내에 객체가 어디에 속한지 알려주지만 Classification을 위한 network에서 특정 object에서 나타나는 특정한 pattern에 대해 score가 학습이 되므로, 다양한 training sample에서 공통적으로 나타나는 scene들에 대해서 score가 낮게 학습이 된다. 
# 
# - 즉 배경과 잘 구분되는 object와 같이 discriminate한 부분에 집중하여 학습이 되므로, sparse하고 blurrly한 정보를 포함하고 있으며, score가 object의 외각영역에 집중되는 경향이 있다. 
# 
# :::{figure-md} markdown-fig
# <img src="pic/weakly3.png" alt="weakly3" class="bg-primary mb-1" width="800px">
# 
# Examples of Class Attention Map (source: Weakly Supervised Object Detection, ECCV Tutorial 2020)
# :::
#         
# - 이를 해결하기 위해 다양한 기법의 weakly supervised learning 기법이 제안되었고, 이를 분류해보면 다음과 같다.
#     
#     (1) Seeded Region Growing
#     
#     (2) Adversarial Erasing
#     
#     (3) Learning Pixel Affinity
#     
#     (4) Sailency or Attention Mechanism
#     
# +++ {"meta": "data"}

# ## Trend
# 

# In[81]:


#@title
# https://medium.com/analytics-vidhya/creating-a-dual-axis-pareto-chart-in-altair-e3673107dd14
# https://altair-viz.github.io/user_guide/interactions.html
# https://www.datacamp.com/tutorial/altair-in-python

import altair as alt
import pandas as pd

y_scale = [50, 90]
x_scale = [0, 7]

viridis = ['#440154', '#472c7a', '#3b518b', '#2c718e', '#21908d', '#27ad81', '#5cc863', '#aadc32', '#fde725']


data = pd.DataFrame({'type': 4*["Full Supervision"],
                     'idx': [1, 2, 3, 4],
                     'year': ["2015", "2017", "2017", "2018"],
                     'nickname': ['FCN', 'PSPNet', 'DeepLab_v3', 'DeepLab_v3+'],
                     'miou' :[67.2, 85.4, 86.9, 89.0],
                     'backbone' :["unknown","unknown","unknown","unknown"]
                     }
                     )

weakly_data = pd.DataFrame({'type': 4*["WS-learning pixel affinity"],
                     'idx': [3, 4, 5, 6],
                     'year': ["2017", "2018", "2019.", "2020"],
                     'nickname': ['RAKW', 'AffinityNet', 'IRNet', 'CIAN'],
                     'miou' :[62.2, 63.7, 64.8, 65.3],
                     'backbone' :["unknown","unknown","unknown","unknown"]
                     }
                     )

weakly2_data = pd.DataFrame({'type': 3*["WS-adversarial erasing"],
                     'idx': [3, 4, 6],
                     'year': ["2017", "2018", "2020"],
                     'nickname': ['AE-PSL', 'SeeNet', 'EADER'],
                     'miou' :[55.7, 62.8, 63.8],
                     'backbone' :["unknown","unknown","unknown"]
                     }
                     )

def GetGraphElement(data, x_scale, y_scale, perf_measure, line_color = "#000000", point_color = "#000000", text_color = "#000000", text_y_pos = -10, textperf_y_pos=-20):
    base = alt.Chart(data).encode(
    x = alt.X("idx", scale=alt.Scale(domain=x_scale),axis=None),
    ).properties (
    width = 800,
    title = ["Trend on mIoU (PASCAL VOC 2012 testset)"]
    )

    line = base.mark_line(strokeWidth= 1.5, color = line_color).encode(
        y=alt.Y('miou', scale=alt.Scale(domain=y_scale),axis=alt.Axis(grid=True)),
        #text = alt.Text('nickname')
        color=alt.Color('type'),
        #opacity='type'
    )

    points = base.mark_circle(strokeWidth= 3, color = point_color).encode(
            y=alt.Y(perf_measure, scale=alt.Scale(domain=y_scale), axis=None),
            tooltip = [alt.Tooltip('year'),
            alt.Tooltip('nickname'),
            alt.Tooltip(perf_measure),
            alt.Tooltip('backbone'),],
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
    

base, line, points, point_nick, point_perf = GetGraphElement(data, x_scale, y_scale, 'miou', 
                                                            line_color = "#fde725", point_color = "#000000", text_color = "#000000", 
                                                            text_y_pos = -10, textperf_y_pos=20)
base2, line2, points2, point_nick2, point_perf2 = GetGraphElement(weakly_data, x_scale, y_scale, 'miou', 
                                                                    line_color = "#cb4154", point_color = "#000000", text_color = "#000000", 
                                                                    text_y_pos = -20, textperf_y_pos=-30)
base3, line3, points3, point_nick3, point_perf3 = GetGraphElement(weakly2_data, x_scale, y_scale, 'miou', 
                                                                    line_color = "#3b518b", point_color = "#000000", text_color = "#000000", 
                                                                    text_y_pos = 20, textperf_y_pos=30)

(
    line+points+point_nick+point_perf+
    line2+points2+point_nick2+point_perf2+ 
    line3+points3+point_nick3+point_perf3
).resolve_scale(y = 'independent')


# 
# +++
# 
#  - 본 chapter에서는 위 분류에 속한 다양한 weakly-supervised learning 기반 segmentation 연구결과를 리뷰한다.
# 
# *Latest update: Jun 21, 2022*
