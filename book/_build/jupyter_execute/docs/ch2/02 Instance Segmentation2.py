#!/usr/bin/env python
# coding: utf-8

# # Overview
# 

# ## Trend
# 

# In[5]:


#@title
# https://medium.com/analytics-vidhya/creating-a-dual-axis-pareto-chart-in-altair-e3673107dd14
# https://altair-viz.github.io/user_guide/interactions.html
# https://www.datacamp.com/tutorial/altair-in-python

import altair as alt
import pandas as pd

y_scale = [25, 60]
x_scale = [0, 14]

data = pd.DataFrame({'type': 13*["Inst. Segmentation"],
                     'idx': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                     'year': ["20xx", "20xx", "20xx", "20xx", "20xx", "20xx", "20xx", "20xx", "20xx", "20xx", "20xx", "20xx", "20xx"],
                     'nickname': ['YOLACT', 'YOLACT++', 'Mask-RCNN', 'CenterMask','HTC','BCNet', 'ISTR', 'D2Det', 'SOLO', 'BlendMask', 'DetectoRS', 'QueryInst', 'Swin-L'],
                     'miou' :[29.8, 34.6, 35.7, 38.3, 39.7, 39.8, 39.9, 40.2, 40.6, 41.3, 47.5, 49.1, 50.2],
                     'backbone' :["unknown","unknown","unknown","unknown", "unknown","unknown","unknown","unknown", "unknown","unknown","unknown", "unknown", "unknown"]
                     }
                     )

def GetGraphElement(data, x_scale, y_scale, perf_measure, line_color = "#000000", point_color = "#000000", text_color = "#000000", text_y_pos = -10, textperf_y_pos=-20):
    base = alt.Chart(data).encode(
    x = alt.X("idx", scale=alt.Scale(domain=x_scale),axis=None),
    ).properties (
    width = 800,
    title = ["Trend on AP (COCO Test-dev)"]
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

(
    line+points+point_nick+point_perf
).resolve_scale(y = 'independent')


# 
# *Latest update: Jan 6, 2022*
