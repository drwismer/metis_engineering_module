import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.db_interaction import DatabaseInteraction
from data.time_series_functions import add_regression_column, get_slope

from bokeh.models import ColumnDataSource, Span, BoxAnnotation, Div, HoverTool, LogAxis, LinearAxis, Range1d, DataRange1d
from bokeh.models.glyphs import Text
from bokeh.models.widgets import DataTable, DateFormatter, NumberFormatter, TableColumn, DateRangeSlider, RadioButtonGroup
from bokeh.io import push_notebook, show, output_notebook, curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, row, widgetbox, Spacer
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.transform import cumsum

from functools import partial
from jesse import research
from scipy.stats import linregress
from math import pi, sin, sqrt, radians

import pickle
from pathlib import Path


def modify_doc(doc, rf_dict, ichimoku_df, candles_df, macd_df, rsi_df, bbands_df, volume_df, glassnode_df, divergence_df):
    """
    Create and display Bokeh dashboard.
    """
    
    def add_tooltips(graph, renderers, tips):
        for r, t in zip(renderers, tips):
            graph.add_tools(HoverTool(renderers=r, tooltips=t, formatters={'@date' : 'datetime'}))
            
    # Tooltip formats
    num_frmt = '{0,0}'
    dec_frmt = '{0.00000000}'
    
    
    # RANDOM FOREST PIE CHARTS
    # --------------------------------------------------------------------------------------------------------------
    
    buy_sell = figure(width=600, height=600, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    
    data = pd.Series(rf_dict['all'][1][0]).reset_index(name='value')
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['color'] = ['#3AD835', '#EA5353']
    data['prediction'] = ['Up', 'Down']
    
    theta = rf_dict['all'][1][0][0] * 360 / 2
    r = 0.75
    
    buy_sell.wedge(x=0, y=0, radius=r,
                   start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                   line_color="white", fill_color='color', legend_field='prediction', source=data)

    buy_sell.axis.axis_label=None
    buy_sell.axis.visible=False
    buy_sell.grid.grid_line_color = None
    
    text = rf_dict['all'][1][0]
    text = ['{:.1%}'.format(x) for x in text]
    
    if theta > 90:
        theta = 180 - theta
        y1 = sin(radians(theta)) * r * 0.5
        x1 = sqrt((r * 0.5)**2 - y1**2) * -1
        text_align=[0, 1]
    else:
        y1 = sin(radians(theta)) * r * 0.5
        x1 = sqrt((r * 0.5)**2 - y1**2)
        text_align=[1, 0]
    
    x2 = x1 * -1
    y2 = y1 * -1
    
    source = ColumnDataSource(dict(x=[x1, x2], y=[y1, y2], text=text))

    glyph = Text(x='x', y='y', text='text', text_color='black', text_font_size='22px')
    buy_sell.add_glyph(source, glyph)
    
    
    # ICHIMOKU GRAPHS
    # --------------------------------------------------------------------------------------------------------------
    
    # Create data sources for Ichimoku indicators, cloud (vertical area fill), and candles
    ichimoku_source = ColumnDataSource(ichimoku_df)
    candle_source_1 = ColumnDataSource(candles_df.loc[candles_df['inc']])
    candle_source_2 = ColumnDataSource(candles_df.loc[candles_df['inc']==False])
    
    # Instantiate graph, hide x-axis, and plot Ichimoku lines
    ichimoku_graph = figure(title="Ichimoku Cloud - BTC", x_axis_type='datetime', width=1000, height=600)
    ichimoku_graph.title.align = 'center'
    ichimoku_graph.xaxis.visible = False
    
    tenkan = ichimoku_graph.line(x='date', y='tenkan', line_width=1, color='#ACFAFE', source=ichimoku_source, name='tenkan')
    kijun = ichimoku_graph.line(x='date', y='kijun', line_width=1, color='#B40C0C', source=ichimoku_source, name='kijun')
    lag_span = ichimoku_graph.line(x='date', y='lagging_span_line', line_width=1, color='#2D8E25', source=ichimoku_source, 
                                   name='lagging_span')
    span_a = ichimoku_graph.line(x='date', y='span_a', line_width=1, color='#2D8E25', source=ichimoku_source, name='span_a')
    span_b = ichimoku_graph.line(x='date', y='span_b', line_width=1, color='#8E2525', source=ichimoku_source, name='span_b')
    
    
    # Plot the cloud (vertical area fill between span lines)
    green_cloud = ichimoku_graph.varea(x='date', y1='span_a', y2='span_b_vertical', fill_color='#8FE54F', fill_alpha=0.2, 
                                       source=ichimoku_source, name='cloud_1')
    red_cloud = ichimoku_graph.varea(x='date', y1='span_a_vertical', y2='span_b', fill_color='#E54F4F', fill_alpha=0.2, 
                                     source=ichimoku_source, name='cloud_2')
    
    
    # Plot candles
    def plot_candles(plot):
        """
        Accept a plot, return candle line segments and vertical bars
        """
        width = 12*60*60*1000
        
        green_wick = plot.segment('date', 'high', 'date', 'low', color='#3AD835', source=candle_source_1, name='green_wick')
        red_wick = plot.segment('date', 'high', 'date', 'low', color='#EA5353', source=candle_source_2, name='red_wick')
        green_candle = plot.vbar(x='date', top='close', bottom='open', width=width, fill_color='#3AD835', line_color='#3AD835',
                                 source=candle_source_1, name='green_candle')
        red_candle = plot.vbar(x='date', top='close', bottom='open', width=width, fill_color='#EA5353', line_color='#EA5353',
                               source=candle_source_2, name='red_candle')
        
        return green_wick, red_wick, green_candle, red_candle
    
    ichi_green_wick, ichi_red_wick, ichi_green_candle, ichi_red_candle = plot_candles(ichimoku_graph)

    # Create date range slider and number or weeks button
    start_date = ichimoku_df['date'].iloc[0]
    end_date = ichimoku_df['date'].iloc[-1]
    
    date_slider = DateRangeSlider(start=start_date, end=end_date, 
                                      value=(start_date, end_date),
                                      format='%Y-%m-%d'
                                     )
    
    num_weeks_radio_labels = ['All', '1', '2', '4', '10', '26', '52', '104', '208']
    num_weeks_button = RadioButtonGroup(labels=num_weeks_radio_labels, active=0)
    
    # Create tooltips for candles, cloud, and Ichimoku lines
    candle_tips = [('Date', '@date{%F}'),
                   ('Open', '@open'+num_frmt),
                   ('Close', '@close'+num_frmt),
                   ('High', '@high'+num_frmt),
                   ('Low', '@low'+num_frmt)
                  ]
    
    cloud_tips = [('Date', '@date{%F}'),
                  ('Span A (Green)', '@span_a'+num_frmt),
                  ('Span B (Red)', '@span_b'+num_frmt)
                 ]
    
    line_tips = [('Date', '@date{%F}'),
                 ('Tenkan', '@tenkan'+num_frmt),
                 ('Kijun', '@kijun'+num_frmt)
                ]
    
    
    candle_renderers = [ichi_green_wick, ichi_red_wick, ichi_green_candle, ichi_red_candle]
    cloud_renderers = [span_a, span_b, green_cloud, red_cloud]
    line_renderers = [tenkan, kijun]
    
    tips = [candle_tips, cloud_tips, line_tips]
    renderers = [candle_renderers, cloud_renderers, line_renderers]
    
    add_tooltips(ichimoku_graph, renderers, tips)
    
    # Create cloud size line graph
    kumo_df = ichimoku_df.copy()
    
    kumo_df.loc[((kumo_df['cloud_over_price'] > 0) & (kumo_df['future_cloud_over_price'] > 0)) |
                ((kumo_df['cloud_over_price'] < 0) & (kumo_df['future_cloud_over_price'] < 0)), 
                ['cloud_over_price','future_cloud_over_price']
               ] = np.nan  
    
    kumo_source_1 = ColumnDataSource(ichimoku_df.loc[(kumo_df['cloud_over_price'] > 0) &
                                                     (kumo_df['future_cloud_over_price'] < 0)
                                                    ]) 
    kumo_source_2 = ColumnDataSource(ichimoku_df.loc[(kumo_df['cloud_over_price'] > 0) & 
                                                     (kumo_df['future_cloud_over_price'] < 0)
                                                    ])
    
    cloud_size_graph = figure(title='Cloud Size vs. Closing Price', x_axis_type='datetime', 
                               width=1000, height=240)
    cloud_size_graph.yaxis.visible = False
    cloud_size_graph.ygrid.visible = False
    cloud_size_graph.title.align = 'center'
    
    kumo_bullish = cloud_size_graph.varea(x='date', y1='future_cloud_over_price', y2='future_cloud_over_price_vertical', 
                                          fill_color='#8FE54F', fill_alpha=.7, source=ichimoku_source, name='kumo_1', 
                                          legend_label='Future Bullish Kumo Twist')
    
    kumo_bearish = cloud_size_graph.varea(x='date', y1='cloud_over_price', y2='cloud_over_price_vertical',
                                          fill_color='#E54F4F', fill_alpha=.7, source=ichimoku_source, name='kumo_2', 
                                          legend_label='Future Bearish Kumo Twist')
    
    kumo_twist = cloud_size_graph.line(x=[], y=[], color='white', line_dash='dashed', line_width=2, legend_label='Cloud Size = 0') 
    
    cloud_size = cloud_size_graph.line(x='date', y='cloud_over_price', color='#00FFFF', line_width=1.5,
                                       source=ichimoku_source, name='cloud_size', legend_label='Current Cloud')
    
    future_cloud_size = cloud_size_graph.line(x='date', y='future_cloud_over_price', color='#FFC100', line_width=1.5, 
                                              source=ichimoku_source, name='cloud_size', legend_label='Future Cloud')
    
    zero_line = Span(location=0, dimension='width', line_color='white', line_dash='dashed', line_width=1, level='underlay')
    bullish = BoxAnnotation(bottom=0, top=1, fill_alpha=0.2, fill_color='#09C400', line_color='#09C400', level='underlay')
    bearish = BoxAnnotation(top=0, bottom=-1, fill_alpha=0.2, fill_color='#C40000', line_color='#C40000', level='underlay')
    
    cloud_size_graph.renderers.extend([zero_line])
    cloud_size_graph.add_layout(bullish)
    cloud_size_graph.add_layout(bearish)
    
    cloud_size_graph.legend.location = 'top_left'
    cloud_size_graph.legend.orientation = 'vertical'
    cloud_size_graph.legend.label_text_font_size = '10px'
    cloud_size_graph.legend.spacing = 10
    cloud_size_graph.legend.margin = 10
    cloud_size_graph.legend.label_text_color = 'white'
    cloud_size_graph.legend.background_fill_color = 'black'
    cloud_size_graph.legend.background_fill_alpha = 0.9
    cloud_size_graph.legend.border_line_color = 'black'   
    cloud_size_graph.legend.border_line_alpha = 1
    
    # Create tooltips for cloud size graph
    cloud_size_tips = [('Date', '@date{%F}'),
                       ('Cloud Size vs. Price', '@cloud_over_price'+'{:.1%}'),
                       ('Future Cloud Size vs. Price', '@future_cloud_over_price'+'{:.1%}')
                      ]
    
    cloud_size_renderers = [cloud_size, future_cloud_size, kumo_twist]
    
    add_tooltips(cloud_size_graph, [cloud_size_renderers], [cloud_size_tips])
    
    
    # --------------------------------------------------------------------------------------------------------------
    
    # MACD
    # --------------------------------------------------------------------------------------------------------------
    # Filter for MACD data to be graphed 
    macd_data = macd_df[['date', 'histogram', 'macd', 'signal']]
    
    # Create MACD data sources
    macd_source = ColumnDataSource(macd_data)
    hist_source_1 = ColumnDataSource(macd_data.loc[(macd_data['histogram'] > 0)])
    hist_source_2 = ColumnDataSource(macd_data.loc[(macd_data['histogram'] < 0)])
    
    # Instantiate MACD figure, hide y-axis and grid, and plot histogram and lines
    macd_graph = figure(title='MACD', title_location='left', x_axis_type='datetime',  
                        width=1000, height=160)
    macd_graph.yaxis.visible = False
    macd_graph.ygrid.visible = False
    macd_graph.title.align = 'center'
    
    hist_pos = macd_graph.vbar(x='date', top='histogram', bottom=0, width=24*60*60*1000*.8, fill_color='#3AD835',
                               line_color='#3AD835',source=hist_source_1, name='hist_pos'
                              )
    hist_neg = macd_graph.vbar(x='date', top=0, bottom='histogram', width=24*60*60*1000*.8, fill_color='#EA5353',
                               line_color='#EA5353', source=hist_source_2, name='hist_neg'
                              )
    macd = macd_graph.line(x='date', y='macd', line_width=1, color='#3CA9FF', source=macd_source, name='macd')
    signal = macd_graph.line(x='date', y='signal', line_width=1, color='#FF9C26', source=macd_source, name='signal')
    
    # --------------------------------------------------------------------------------------------------------------
    
    # BOLLINGER BANDS
    # --------------------------------------------------------------------------------------------------------------
    # Filter for Bollinger Bands data to be graphed
    bbands_data = bbands_df.copy()
    bbands_source = ColumnDataSource(bbands_data)
    
    # Instantiate graph, hide x-axis, and plot Bollinger Bands
    bbands_graph = figure(title='Bollinger Bands - BTC', x_axis_type='datetime',  
                          width=1000, height=600)
    bbands_graph.xaxis.visible = False
    
    upper = bbands_graph.line(x='date', y='upper', line_width=1, color='#9B37FF', source=bbands_source, name='upper')
    lower = bbands_graph.line(x='date', y='lower', line_width=1, color='#9B37FF', source=bbands_source, name='lower')
    middle = bbands_graph.line(x='date', y='middle', line_width=1, color='#FF9E37', source=bbands_source, name='middle')
    band = bbands_graph.varea(x='date', y1='upper', y2='lower', fill_color='#E9BAFF', fill_alpha=0.2, 
                              source=bbands_source, name='band')
    
    # Plot candles
    bbands_green_wick, bbands_red_wick, bbands_green_candle, bbands_red_candle = plot_candles(bbands_graph)
    
    # Create tooltips for candles and Bollinger Bands (use same candles tips from Ichimoku graph)
    bbands_tips = [('Date', '@date{%F}'),
                   ('Upper Band', '@upper'+num_frmt),
                   ('Lower Band', '@lower'+num_frmt),
                   ('Middle', '@middle'+num_frmt)
                  ]
    
    candle_renderers = [bbands_green_wick, bbands_red_wick, bbands_green_candle, bbands_red_candle]
    bbands_renderers = [upper, lower, middle]
    
    tips = [candle_tips, bbands_tips]
    renderers = [candle_renderers, bbands_renderers]
    
    add_tooltips(bbands_graph, renderers, tips)
    
    # Create band size line graph
    bbands_size_graph = figure(title='Bollinger Band Size vs. Closing Price', x_axis_type='datetime', 
                               width=1000, height=240)
    bbands_size_graph.yaxis.visible = False
    bbands_size_graph.ygrid.visible = False
    bbands_size_graph.title.align = 'center'
    
    bbands_size = bbands_size_graph.line(x='date', y='band_size_vs_closing_price', color='white', 
                                         source=bbands_source, name='band_size', legend_label='Band Size')
    
    # Add quartile lines
    q1 = bbands_data['band_size_vs_closing_price'].quantile(.25)
    q2 = bbands_data['band_size_vs_closing_price'].quantile(.5)
    q3 = bbands_data['band_size_vs_closing_price'].quantile(.75)
    
    q1_line = bbands_size_graph.line(x='date', y=q1, color='#FF5A5A', line_dash='dotted', source=bbands_source, 
                                     name='q1', legend_label='Q1')
    q2_line = bbands_size_graph.line(x='date', y=q2, color='#86C8FF', line_dash='dotted', source=bbands_source, 
                                     name='q2', legend_label='Q2')
    q3_line = bbands_size_graph.line(x='date', y=q3, color='#D386FF', line_dash='dotted', source=bbands_source, 
                                     name='q3', legend_label='Q3')
    
    bbands_size_graph.legend.location = 'top_left'
    bbands_size_graph.legend.orientation = 'horizontal'
    bbands_size_graph.legend.label_text_font_size = '10px'
    bbands_size_graph.legend.spacing = 10
    bbands_size_graph.legend.margin = 10
    bbands_size_graph.legend.label_text_color = 'white'
    bbands_size_graph.legend.background_fill_color = 'black'
    bbands_size_graph.legend.background_fill_alpha = 0.9
    bbands_size_graph.legend.border_line_color = 'black'   
    bbands_size_graph.legend.border_line_alpha = 1
    
    # Create tooltips for band size graph
    bbands_size_tips = [('Date', '@date{%F}'),
                        ('Band Size vs. Close', '@band_size_vs_closing_price'+'{:.1%}'),
                        ('1st Quartile', '{:.1%}'.format(q1)),
                        ('2nd Quartile', '{:.1%}'.format(q2)),
                        ('3rd Quartile', '{:.1%}'.format(q3))
                       ]
    
    bbands_size_renderers = [bbands_size, q1_line, q2_line, q3_line]
    
    add_tooltips(bbands_size_graph, [bbands_size_renderers], [bbands_size_tips])
    
    
    # Bollinger Band position graph
    bbands_position_graph = figure(title='Price Position in Bollinger Bands', x_axis_type='datetime',  width=1000, height=240,
                                  y_range = (-0.3, 1.3))
    bbands_position_graph.yaxis.visible = False
    bbands_position_graph.ygrid.visible = False
    bbands_position_graph.title.align = 'center'
    
    bbands_position = bbands_position_graph.line(x='date', y='closing_price_band_position', color='#7EFFFD', 
                                                 source=bbands_source, name='band_position', legend_label='Price Position')
    
    # Set default moving average line and plot
    bbands_source.data['band_position_avg'] = bbands_source.data['band_position15']
    bbands_position_avg = bbands_position_graph.line(x='date', y='band_position_avg', color='#FF7E7E', line_width=1.5,
                                                     source=bbands_source, name='band_position_avg', legend_label='Moving Average')
    
    # Plot band lines
    perc_100 = bbands_position_graph.line(x='date', y=1, color='#9B37FF', line_width=2, source=bbands_source, name='upper', 
                                          legend_label='Upper')
    perc_50 = bbands_position_graph.line(x='date', y=0.5, color='#FF9E37', line_width=2, source=bbands_source, 
                                         name='middle', legend_label='Middle')
    perc_0 = bbands_position_graph.line(x='date', y=0, color='#9B37FF', line_width=2, source=bbands_source, name='lower',
                                        legend_label='Lower')
    
    # Create button options
    bbands_radio_labels = ['3', '7', '15', '30', '60', '120']
    bbands_button = RadioButtonGroup(labels=bbands_radio_labels, active=2)
    
    def bbands_period_change(selection):#attr, old, new):
        bbands_source.data['band_position_avg'] = bbands_source.data['band_position' +
                                                                     str(int(bbands_radio_labels[selection]))
                                                                    ]
    
    # Stylize axes and legend
    bbands_graph.yaxis.visible = False
    bbands_graph.ygrid.visible = False
    #bbands_graph.xaxis.visible = 
    bbands_graph.title.align = 'center'

    bbands_position_graph.legend.location = 'top_left'
    bbands_position_graph.legend.orientation = 'vertical'
    bbands_position_graph.legend.label_text_font_size = '10px'
    bbands_position_graph.legend.spacing = 10
    bbands_position_graph.legend.margin = 10
    bbands_position_graph.legend.label_text_color = 'white'
    bbands_position_graph.legend.background_fill_color = 'black'
    bbands_position_graph.legend.background_fill_alpha = 0.9
    bbands_position_graph.legend.border_line_color = 'black'   
    bbands_position_graph.legend.border_line_alpha = 1
    
    # Create tooltips for band position graph
    bbands_position_tips = [('Date', '@date{%F}'),
                            ('Price Position', '@closing_price_band_position'+'{:.1%}'),
                            ('Moving Average', '@band_position_avg'+'{:.1%}')
                           ]
    
    bbands_position_renderers = [bbands_position, bbands_position_avg]
    
    add_tooltips(bbands_position_graph, [bbands_position_renderers], [bbands_position_tips])
        
    
    # --------------------------------------------------------------------------------------------------------------
    
    # RSI
    # --------------------------------------------------------------------------------------------------------------
    # Filter for RSI data to be graphed 
    rsi_data = rsi_df[['date', 'rsi']]
    
    # Create RSI data sources
    rsi_source = ColumnDataSource(rsi_data)
    
    # Instantiate RSI figure, hide y-axis and grid, and plot lines
    rsi_graph = figure(title='RSI', title_location='left', x_axis_type='datetime', 
                       y_range=(0, 100), width=1000, height=160)
    rsi_graph.yaxis.visible = False
    rsi_graph.ygrid.visible = False
    rsi_graph.title.align = 'center'
    
    rsi = rsi_graph.line(x='date', y='rsi', color='#9A21D3', source=rsi_source, name='rsi')
    
    over = 70
    under = 30
    
    oversold = Span(location=over, dimension='width', line_color='white', line_dash='dashed', line_width=1)
    safe_zone = BoxAnnotation(bottom=under, top=over, fill_alpha=0.2, fill_color='#E9BAFF')
    undersold = Span(location=under, dimension='width', line_color='white', line_dash='dashed', line_width=1)
    
    rsi_graph.renderers.extend([oversold, undersold])
    rsi_graph.add_layout(safe_zone)
    
    # Create tooltips for RSI
    num_frmt = '{0,0}'
    
    rsi_tips = [('Date', '@date{%F}'),
                 ('RSI', '@rsi'+num_frmt),
                ]
    
    rsi_renderers = [rsi]
    
    tips = [rsi_tips]
    renderers = [rsi_renderers]
    
    add_tooltips(rsi_graph, renderers, tips)
    
    # --------------------------------------------------------------------------------------------------------------
    
    # VOLUME
    # --------------------------------------------------------------------------------------------------------------
    # Filter for volume data to be graphed 
    vol_data = volume_df[['date', 'volume', 'inc']]
    
    # Create volume data sources
    vol_source_1 = ColumnDataSource(vol_data.loc[vol_data['inc']])
    vol_source_2 = ColumnDataSource(vol_data.loc[vol_data['inc']==False])
    
    # Instantiate volume figure, hide axes and grid, and plot bars
    vol_graph = figure(title='Volume', title_location='left', x_axis_type='datetime', 
                       width=1000, height=140)
    vol_graph.xaxis.visible = False
    vol_graph.yaxis.visible = False
    vol_graph.ygrid.visible = False
    vol_graph.title.align = 'center'
    
    vol_green = vol_graph.vbar(x='date', top='volume', bottom=0, width=24*60*60*1000*.8, fill_color='#3AD835',
                               line_color='#3AD835',source=vol_source_1, name='vol_green'
                              )
    vol_red = vol_graph.vbar(x='date', top='volume', bottom=0, width=24*60*60*1000*.8, fill_color='#EA5353',
                               line_color='#EA5353',source=vol_source_2, name='vol_red'
                              )
    
    # Create tooltips for volume
    num_frmt = '{0,0}'
    
    vol_tips = [('Date', '@date{%F}'),
                 ('Volume', '@volume'+num_frmt),
                ]
    
    vol_renderers = [vol_green, vol_red]
    
    tips = [vol_tips]
    renderers = [vol_renderers]
    
    add_tooltips(vol_graph, renderers, tips)
    
    # --------------------------------------------------------------------------------------------------------------
    
    # ON-CHAIN
    # --------------------------------------------------------------------------------------------------------------
    # Filter for RSI data to be graphed 
    gn_data = glassnode_df.copy(deep=True)
    
    color_dict = {'red' : '#FF7E7E',
                  'orange' : '#FFD07E',
                  'yellow' : '#FBFF7E',
                  'green' : '#92FF7E',
                  'blue' : '#7EFFFD'
                 }
    
    gn_dict = {'NUPL' : {'Capitulation (<0)' : [None, 0.0, color_dict['red']],
                         'Hope/Fear (0-0.25)' : [0.0, .25, color_dict['orange']],
                         'Optimism/Anxiety (0.25-0.5)' : [.25, .5, color_dict['yellow']],
                         'Belief/Denial (0.5-0.75)' : [.5, .75, color_dict['green']],
                         'Euphoria/Greed (>0.75)' : [.75, None, color_dict['blue']]
                        },
               'STFD' : {'Less Than 1' : [None, 1.0, color_dict['red']],
                         'Greater Than 1' : [1.0, None, color_dict['blue']]
                        },
               'M/TC' : {'0-0.000001' : [None, 0.000001, color_dict['red']],
                         '0.000001-0.000002' : [0.000001, .000002, color_dict['orange']],
                         '0.000002-0.000003' : [.000002, .000003, color_dict['yellow']],
                         '0.000003-0.000004' : [.000003, .000004, color_dict['green']],
                         '>0.000004' : [.000004, None, color_dict['blue']]
                        },
               'PSP' : {'0-50%' : [None, .5, color_dict['red']],
                        '50-65%' : [.5, .65, color_dict['orange']],
                        '65-80%' : [.65, .8, color_dict['yellow']],
                        '80-95%' : [.8, .95, color_dict['green']],
                        '>95%' : [.95, None, color_dict['blue']]
                       },
               'Puell' : {'<0.5' : [None, .5, color_dict['red']],
                          '0.5-1.0' : [.5, 1.0, color_dict['orange']],
                          '1.0-2.5' : [1.0, 2.5, color_dict['yellow']],
                          '2.5-4.0' : [2.5, 4.0, color_dict['green']],
                          '>4.0' : [4.0, None, color_dict['blue']]
                         }
              }
    
    gn_graph_to_column = {'NUPL' : 'nupl',
                          'STFD' : 'stfd',
                          'M/TC' : 'market_to_thermo',
                          'PSP' : 'perc_supply_profit',
                          'Puell' : 'puell'
                         }
    
    # Establish data source for closing price line (use volume df) and on-chain data
    gn_data = volume_df[['date', 'closing_price']]
    gn_data = gn_data.merge(glassnode_df,how='left')
    gn_source = ColumnDataSource(gn_data)
    
    def plot_glassnode(key, show_x=False):
        if show_x:
            height = 80
        else:
            height = 60
        
        gn_graph = figure(title=key, title_location='left', x_axis_type='datetime', y_axis_type='log',
                          width=1000, height=160)
        metric_graph = figure(x_axis_type='datetime', y_axis_type='log',width=1000, height=height)
        
        gn_graph.yaxis.visible = False
        gn_graph.ygrid.visible = False
        gn_graph.xaxis.visible = False
        gn_graph.title.align = 'center'
        
        metric_graph.yaxis.visible = False
        metric_graph.ygrid.visible = False
        metric_graph.xaxis.visible = show_x

        col_name = gn_graph_to_column[key]
        
        price = gn_graph.line(x='date', y='closing_price', line_width=1, color='black', source=gn_source, 
                              legend_label='Price', name='price')

        metric = metric_graph.line(x='date', y=col_name, line_width=1, color='white', source=gn_source, name='metric')
        
        for key2 in gn_dict[key]:
            lower = gn_dict[key][key2][0]
            upper = gn_dict[key][key2][1]
            color = gn_dict[key][key2][2]
            
            gn_subset = gn_data[['date', col_name]]

            if (type(lower) == float) & (type(upper) == float):
                conditions = (gn_data[col_name] > lower) & (gn_data[col_name] <= upper)
            elif type(lower) == float:
                conditions = (gn_data[col_name] > lower)
            elif type(upper) == float:
                conditions = (gn_data[col_name] <= upper)
            else:
                conditions = None

            gn_subset['in_range'] = conditions
            
            gn_graph.line([], [], legend_label=key2, line_width=10, line_color=color)
            
            try:
                gn_subset['colored'] = (gn_subset['in_range'].diff(1) != 0).astype('int').cumsum()
                
                min_date_df = gn_subset[gn_subset.groupby('colored').date.transform('min') == gn_subset['date']]
                max_date_df = gn_subset[gn_subset.groupby('colored').date.transform('min') == gn_subset['date']]

                start = min_date_df[min_date_df['in_range']]['colored'].min()
                
                for i in range(start, gn_subset['colored'].max() + 1, 2):
                    min_date = gn_subset['date'].iloc[gn_subset['colored'].values.searchsorted(i, side='left')]
                    max_date = gn_subset['date'].iloc[gn_subset['colored'].values.searchsorted(i, side='right') - 1]

                    box = BoxAnnotation(left=min_date, right=max_date + timedelta(days=1), fill_color=color, fill_alpha=1,
                                        line_color=color, level='underlay')
                    gn_graph.add_layout(box)
                
            except:
                pass
        
        gn_graph.legend.location = 'top_left'
        gn_graph.legend.orientation = 'vertical'
        gn_graph.legend.label_text_font_size = '10px'
        gn_graph.legend.spacing = 0
        gn_graph.legend.margin = 5
        gn_graph.legend.label_text_color = 'black'
        gn_graph.legend.background_fill_color = 'white'
        gn_graph.legend.background_fill_alpha = 0.75
        gn_graph.legend.border_line_color = 'black'   
        gn_graph.legend.border_line_alpha = 1
        
        # Create tooltips for on-chain
        num_frmt = '{0,0}'

        gn_tips = [('Date', '@date{%F}'),
                   ('Price', '@closing_price'+num_frmt),
                   (key, '@'+gn_graph_to_column[key]+dec_frmt)
                  ]

        gn_renderers = [price]
        metric_renderers = [metric]

        tips = [gn_tips]

        add_tooltips(gn_graph, [gn_renderers], tips)
        add_tooltips(metric_graph, [metric_renderers], tips)
        
        return gn_graph, metric_graph
    
    nupl_vs_price, nupl_graph = plot_glassnode('NUPL')
    stfd_vs_price, stfd_graph = plot_glassnode('STFD')
    mtc_vs_price, mtc_graph = plot_glassnode('M/TC')
    psp_vs_price, psp_graph = plot_glassnode('PSP')
    puell_vs_price, puell_graph = plot_glassnode('Puell', show_x=True)
    
    # --------------------------------------------------------------------------------------------------------------
    
    # DIVERGENCES
    # --------------------------------------------------------------------------------------------------------------

    div_data = divergence_df.copy(deep=True)
    
    # Set defaults
    div_data['close_reg_norm'] = div_data['close_reg_norm15']
    div_data['volume_reg_norm'] = div_data['volume_reg_norm15']
    div_data['macd_reg_norm'] = div_data['macd_reg_norm15']
    div_data['rsi_reg_norm'] = div_data['rsi_reg_norm15']
    
    div_source = ColumnDataSource(div_data)
    
    
    def plot_divergence(col_1, col_2, window, metric, col_2_max=None, show_x=False):
        div_source.data[col_1 + '_reg_norm'] = div_source.data[col_1 + '_reg_norm' + str(window)]
        div_source.data[col_2 + '_reg_norm'] = div_source.data[col_2 + '_reg_norm' + str(window)]
        
        col_1_max = max(abs(div_source.data[col_1 + '_reg_norm'].min()), 
                        div_source.data[col_1 + '_reg_norm'].max()
                       )        

        if not col_2_max:
            col_2_max = max(abs(div_source.data[col_2 + '_reg_norm'].min()), 
                            div_source.data[col_2 + '_reg_norm'].max()
                           )

        div_graph = figure(y_range = (col_1_max * -1, col_1_max), width=1000, height=160, x_axis_type='datetime', 
                           title='Price vs. ' + metric, title_location='left')
        
        price = div_graph.line(x='date', y=col_1 + '_reg_norm', color = '#7EFFFD', legend_label='Price', source=div_source)
        div_graph.extra_y_ranges = {'y2': Range1d(col_2_max * -1, col_2_max)}
        div_graph.add_layout(LinearAxis(y_range_name = 'y2'), 'right')

        metric_line = div_graph.line(x='date', y=col_2 + '_reg_norm', color = '#FF7E7E', y_range_name='y2', 
                                legend_label=metric, source=div_source)
        zero_line = Span(location=0, dimension='width', line_color='white', line_dash='dashed', line_width=2)
        div_graph.add_layout(zero_line)
        
        # Stylize axes and legend
        div_graph.yaxis.visible = False
        div_graph.ygrid.visible = False
        div_graph.xaxis.visible = show_x
        div_graph.title.align = 'center'
        
        div_graph.legend.location = 'top_left'
        div_graph.legend.orientation = 'horizontal'
        div_graph.legend.label_text_font_size = '10px'
        div_graph.legend.spacing = 10
        div_graph.legend.margin = 10
        div_graph.legend.label_text_color = 'white'
        div_graph.legend.background_fill_color = 'black'
        div_graph.legend.background_fill_alpha = 0.9
        div_graph.legend.border_line_color = 'black'   
        div_graph.legend.border_line_alpha = 1

        
        # Create tooltips for divergences
        div_tips = [('Date', '@date{%F}'),
                    ('Price', '@close_reg_norm'+dec_frmt),
                    (metric, '@'+col_2+'_reg_norm'+dec_frmt)
                   ]

        div_renderers = [price, metric_line]

        add_tooltips(div_graph, [div_renderers], [div_tips])
        
        return div_graph
    
    vol_price_div = plot_divergence('close', 'volume', 15, 'Volume')
    rsi_price_div = plot_divergence('close', 'rsi', 15, 'RSI')
    macd_price_div = plot_divergence('close', 'macd', 15, 'MACD',col_2_max=0.2, show_x=True)
    
    # Create price and divergence graph
    price_and_div_graph = figure(title='Price and Divergence', x_axis_type='datetime',width=1000, height=600)
    
    div_price = price_and_div_graph.line(x='date', y='close', line_width=1, color='white', source=div_source, 
                                     legend_label='Price', name='price')
        
    price_and_div_graph.yaxis.visible = False
    price_and_div_graph.ygrid.visible = False
    price_and_div_graph.xaxis.visible = False
    price_and_div_graph.title.align = 'center'
    
    # Set up defaults
    price_and_div_data = divergence_df.copy(deep=True)
    price_and_div_data['zeros'] = 0
    price_and_div_source = ColumnDataSource(price_and_div_data)
    
    def set_price_div_data(metric, window, percentile):
        price_and_div_source.data['bullish_div']  = np.where((price_and_div_source.data[metric + '_reg_norm' + str(window) + 
                                                                                        '_perc'] < (percentile)) &
                                                             (price_and_div_source.data['close_reg_norm' + str(window) 
                                                                                 + '_perc'] > (1.0 - percentile)) &
                                                             (price_and_div_source.data[metric + '_reg_norm' + str(window)] > 0) &
                                                             (price_and_div_source.data['close_reg_norm' + str(window)] < 0),
                                                             price_and_div_source.data['close'], 0)
        
        price_and_div_source.data['bearish_div']  = np.where((price_and_div_source.data[metric + '_reg_norm' + str(window) + 
                                                                                        '_perc'] > (1.0 - percentile)) &
                                                             (price_and_div_source.data['close_reg_norm' + str(window) 
                                                                                        + '_perc'] < (percentile)) &
                                                             (price_and_div_source.data[metric + '_reg_norm' + str(window)] < 0) &
                                                             (price_and_div_source.data['close_reg_norm' + str(window)] > 0), 
                                                             price_and_div_source.data['close'], 0)
    
    set_price_div_data('volume', 15, 1.0)

    div_bullish = price_and_div_graph.varea(x='date', y1='bullish_div', y2='zeros', fill_color='#8FE54F', fill_alpha=.7, 
                                            source=price_and_div_source, legend_label='Bullish Divergence', name='div_bullish')
    div_bearish = price_and_div_graph.varea(x='date', y1='bearish_div', y2='zeros', fill_color='#E54F4F', fill_alpha=.7, 
                                            source=price_and_div_source, legend_label='Bearish Divergence', name='div_bearish')
    
    price_and_div_graph.legend.location = 'top_left'
    price_and_div_graph.legend.orientation = 'vertical'
    price_and_div_graph.legend.label_text_font_size = '10px'
    price_and_div_graph.legend.spacing = 10
    price_and_div_graph.legend.margin = 10
    price_and_div_graph.legend.label_text_color = 'white'
    price_and_div_graph.legend.background_fill_color = 'black'
    price_and_div_graph.legend.background_fill_alpha = 0.9
    price_and_div_graph.legend.border_line_color = 'black'   
    price_and_div_graph.legend.border_line_alpha = 1
    
    # Create buttons
    period_radio_labels = ['3', '7', '15', '30', '60', '120']
    period_button = RadioButtonGroup(labels=period_radio_labels, active=2)
    
    metric_radio_labels = ['Volume', 'RSI', 'MACD']
    metric_button = RadioButtonGroup(labels=metric_radio_labels, active=0)
    
    percentile_radio_labels = ['60', '70', '80', '90', '100']
    percentile_button = RadioButtonGroup(labels=percentile_radio_labels, active=4)   
    
    
    def div_update_source_and_range(div_graph, col_1, col_2, window, col_2_max=None):
        div_source.data[col_1 + '_reg_norm'] = div_source.data[col_1 + '_reg_norm' + str(window)]
        div_source.data[col_2 + '_reg_norm'] = div_source.data[col_2 + '_reg_norm' + str(window)]
        
        col_1_max = max(abs(div_source.data[col_1 + '_reg_norm'].min()), 
                        div_source.data[col_1 + '_reg_norm'].max()
                       )        

        if not col_2_max:
            col_2_max = max(abs(div_source.data[col_2 + '_reg_norm'].min()), 
                            div_source.data[col_2 + '_reg_norm'].max()
                           )
        
        div_graph.y_range.start = col_1_max 
        div_graph.y_range.end = col_1_max * -1
        div_graph.extra_y_ranges['y2'].start = col_2_max
        div_graph.extra_y_ranges['y2'].end = col_2_max * -1
    
    def div_period_change(selection):
        div_update_source_and_range(vol_price_div, 'close', 'volume', int(period_radio_labels[selection]))
        div_update_source_and_range(rsi_price_div, 'close', 'rsi', int(period_radio_labels[selection]))
        div_update_source_and_range(macd_price_div, 'close', 'macd', int(period_radio_labels[selection]), col_2_max=0.2)
        
        set_price_div_data(metric_radio_labels[metric_button.active].lower(),
                           int(period_radio_labels[selection]),
                           int(percentile_radio_labels[percentile_button.active])/100
                          )
    
    def div_metric_change(selection):
        set_price_div_data(metric_radio_labels[selection].lower(),
                           int(period_radio_labels[period_button.active]),
                           int(percentile_radio_labels[percentile_button.active])/100
                          )
        
    def div_percentile_change(selection):
        set_price_div_data(metric_radio_labels[metric_button.active].lower(),
                           int(period_radio_labels[period_button.active]),
                           int(percentile_radio_labels[selection])/100
                          )
    
    # --------------------------------------------------------------------------------------------------------------
    
    def callback_date(attr, old, new):
        """
        Adjust graphs and tables based on user date selection.
        """
        start = datetime.fromtimestamp(new[0]/1000).date()
        end = datetime.fromtimestamp(new[1]/1000).date()
        
        # Ichimoku and candle sources
        ichimoku_date_conditions = (ichimoku_df['date'] >= start) & (ichimoku_df['date'] <= end)
        candle_date_conditions = (candles_df['date'] >= start) & (candles_df['date'] <= end)
        
        ichimoku_source.data = ichimoku_df.loc[ichimoku_date_conditions]
        
        candle_source_1.data = candles_df.loc[candle_date_conditions & (candles_df['inc'])]
        
        candle_source_2.data = candles_df.loc[candle_date_conditions & (candles_df['inc']==False)]
        
        # Volume sources
        vol_date_conditions = (vol_data['date'] >= start) & (vol_data['date'] <= end)
        
        vol_source_1.data = vol_data.loc[vol_date_conditions & (vol_data['inc'])]
        vol_source_2.data = vol_data.loc[vol_date_conditions & (vol_data['inc']==False)]
        
        # MACD sources
        macd_date_conditions = (macd_df['date'] >= start) & (macd_df['date'] <= end)
        
        macd_source.data = macd_data.loc[macd_date_conditions]
        hist_source_1.data = macd_df.loc[macd_date_conditions & (macd_data['histogram'] > 0)]
        hist_source_2.data = macd_df.loc[macd_date_conditions & (macd_data['histogram'] < 0)]
        
        # Bollinger Bands source
        bbands_date_conditions = (bbands_df['date'] >= start) & (bbands_df['date'] <= end)
        bbands_source.data = bbands_data.loc[bbands_date_conditions]
        moving_avg = bbands_button.active
        bbands_data['band_position_avg'] = bbands_data['band_position' + bbands_radio_labels[moving_avg]]
        
        # RSI source
        rsi_date_conditions = (rsi_df['date'] >= start) & (rsi_df['date'] <= end)
        rsi_source.data = rsi_data.loc[rsi_date_conditions]
        
        # On-Chain sources
        gn_date_conditions = (gn_data['date'] >= start) & (gn_data['date'] <= end)
        gn_source.data = gn_data.loc[gn_date_conditions]
        
        # Divergence sources
        #div_metric = metric_button.active
        lookback = period_radio_labels[period_button.active]
        #percentile = percentile_button.active
        
        div_data['close_reg_norm'] = div_data['close_reg_norm' + lookback]
        div_data['volume_reg_norm'] = div_data['volume_reg_norm' + lookback]
        div_data['macd_reg_norm'] = div_data['macd_reg_norm' + lookback]
        div_data['rsi_reg_norm'] = div_data['rsi_reg_norm' + lookback]
        
        div_date_conditions = (div_data['date'] >= start) & (div_data['date'] <= end)
        div_source.data = div_data.loc[div_date_conditions]
        
        price_and_div_date_conditions = (price_and_div_data['date'] >= start) & (price_and_div_data['date'] <= end)
        price_and_div_source.data = price_and_div_data.loc[price_and_div_date_conditions]
    
    def callback_weeks(selection):
        """
        Adjust date_slider based on selection.
        """
        if num_weeks_radio_labels[selection] == 'All':
            date_slider.value=(date_slider.start, date_slider.value[1])
        else:
            start = start = datetime.fromtimestamp(date_slider.value[1]/1000).date() - timedelta(days=7 * int(num_weeks_radio_labels[selection]))
            date_slider.value = (start, date_slider.value[1])

    
    date_slider.on_change('value', callback_date)
    num_weeks_button.on_click(callback_weeks)
    period_button.on_click(div_period_change)
    metric_button.on_click(div_metric_change)
    percentile_button.on_click(div_percentile_change)
    bbands_button.on_click(bbands_period_change)  
    
                                                                    
    layout = row(column(Spacer(width=50)),
                 column(Div(text='<h1>One Week Price Prediction - Random Forest<h1>'), buy_sell),
                 column(Spacer(width=50)),
                 column(Div(text='<h1>Ichimoku Cloud<h1>'), ichimoku_graph, vol_graph, macd_graph, date_slider,
                        row(Div(text='<h4>Number of Weeks:   <h4>'), num_weeks_button),
                        cloud_size_graph),
                 column(Spacer(width=50)),
                 column(Div(text='<h1>Bollinger Bands<h1>'), bbands_graph, vol_graph, rsi_graph, date_slider,
                        row(Div(text='<h4>Number of Weeks:   <h4>'), num_weeks_button),
                        bbands_size_graph, bbands_position_graph, 
                        row(Div(text='<h4>Moving Avg Periods:   <h4>'), widgetbox(bbands_button))),
                 column(Spacer(width=50)),
                 column(Div(text='<h1>On-Chain Metrics<h1>'), nupl_vs_price, nupl_graph, stfd_vs_price, stfd_graph, 
                        mtc_vs_price, mtc_graph, psp_vs_price, psp_graph, puell_vs_price, puell_graph, date_slider,
                        row(Div(text='<h4>Number of Weeks:   <h4>'), num_weeks_button)),
                 column(Spacer(width=50)),
                 column(Div(text='<h1>Divergences - Normalized Slope of X Periods Regression<h1>'), 
                        row(Div(text='<h4>Lookback Periods:   <h4>'), widgetbox(period_button)),
                        vol_price_div, rsi_price_div, macd_price_div, date_slider, 
                        row(Div(text='<h4>Number of Weeks:   <h4>'), num_weeks_button),
                        price_and_div_graph,
                        row(Div(text='<h4>Metric Selection:   <h4>'), widgetbox(metric_button)),
                        row(Div(text='<h4>Lookback Periods:   <h4>'), widgetbox(period_button)),
                        row(Div(text='<h4>Top X Percentile:   <h4>'), widgetbox(percentile_button)),
                       ),
                 column(Spacer(width=50)),
                 column(Div(text='<h1>One Week Price Prediction - Random Forest<h1>'), buy_sell),
                 column(Spacer(width=50)),
                 column(Div(text='<h1>Ichimoku Cloud<h1>'), ichimoku_graph, vol_graph, macd_graph, date_slider,
                        row(Div(text='<h4>Number of Weeks:   <h4>'), num_weeks_button),
                        cloud_size_graph),
                 column(Spacer(width=50)),
                 column(Div(text='<h1>Bollinger Bands<h1>'), bbands_graph, vol_graph, rsi_graph, date_slider,
                        row(Div(text='<h4>Number of Weeks:   <h4>'), num_weeks_button),
                        bbands_size_graph, bbands_position_graph, 
                        row(Div(text='<h4>Moving Avg Periods:   <h4>'), widgetbox(bbands_button)))
                )
    
    doc.theme = 'dark_minimal'
    doc.add_root(layout)
    

def main():
    ### Import Data

    # Perform db queries and save to df's.
    DI = DatabaseInteraction()

    glassnode_df = DI.query_to_df(DI.query_all_data('glassnode'))
    ichimoku_df = DI.query_to_df(DI.query_all_data('ichimoku'))
    macd_df = DI.query_to_df(DI.query_all_data('macd'))
    rsi_df = DI.query_to_df(DI.query_all_data('rsi'))
    bbands_df = DI.query_to_df(DI.query_all_data('bbands'))
    volume_df = DI.query_to_df(DI.query_all_data('volume'))

    #Query db for candle data. Only get data for range captured for Ichimoku.
    ichimoku_min_date = ichimoku_df['date'].min()
    ichimoku_max_date = ichimoku_df['date'].max()
    start_date = ichimoku_min_date.strftime('%Y-%m-%d')
    timestamp = DI.query_latest('candle', 'timestamp')/1000

    exchange ='Coinbase'
    coin = 'BTC-USD'
    period = '1D'
    end = datetime.strftime(datetime.fromtimestamp(timestamp), '%Y-%m-%d')

    research.init()
    candles_array = research.get_candles(exchange, coin, period, start_date, end)
    candles_df = pd.DataFrame(candles_array, columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume'])

    # Create date column from timestamp column, add increasing column to signal green and red candles
    candles_df['date'] = candles_df.apply(lambda row: datetime.fromtimestamp(row['timestamp'] / 1000).date(), axis=1)
    candles_df['inc'] = candles_df.close >= candles_df.open

    ### Maniupulate Data for Bokeh Visualizations

    # Convert date column from timestamp to date for glassnode table
    glassnode_df['date'] = glassnode_df['date'].apply(lambda x: x.date())

    # Create df for regression and divergence calculations
    divergence_df = candles_df[['date', 'close', 'volume']].merge(macd_df[['date', 'macd']], 
                                                                  how='left',
                                                                  on='date'
                                                                 )

    divergence_df = divergence_df.merge(rsi_df[['date', 'rsi']],
                                        how='left',
                                        on='date'
                                       )

    # Add regression columns to divergence_df for price (close), volume, RSI, and MACD for each rolling window size.
    windows = [3, 7, 15, 30, 60, 120]    

    for window in windows:
        divergence_df = add_regression_column(divergence_df, 'close', window)
        divergence_df = add_regression_column(divergence_df, 'volume', window)
        divergence_df = add_regression_column(divergence_df, 'rsi', window)
        divergence_df = add_regression_column(divergence_df, 'macd', window)

    divergence_df = divergence_df.dropna(axis=0)

    # Add lagging span and future span details for the graph
    shift = 30
    ichimoku_df['lagging_span_line'] = ichimoku_df['closing_price'].shift(shift * -1)

    i = 1

    while i <= shift:
        new_date = ichimoku_max_date + timedelta(days=i)
        shift_date = new_date - timedelta(days=shift)
        span_a = ichimoku_df.loc[ichimoku_df['date'] == shift_date, 'future_span_a'].iloc[0]
        span_b = ichimoku_df.loc[ichimoku_df['date'] == shift_date, 'future_span_b'].iloc[0]

        ichimoku_df = ichimoku_df.append({'date' : new_date, 'span_a' : span_a, 'span_b' : span_b}, ignore_index=True)

        i += 1


    # Add increasing column to the volume dataframe and the candles dataframe
    volume_df['inc'] = volume_df['closing_price'] > volume_df['closing_price'].shift(1)
    candles_df['inc'] = candles_df.close >= candles_df.open

    # Flip cloud size to negative when span_a < span_b and future_span_a < future_span_b
    ichimoku_df.loc[ichimoku_df['span_a'] < ichimoku_df['span_b'], ['cloud_over_price']] = ichimoku_df['cloud_over_price'] *-1
    ichimoku_df.loc[ichimoku_df['future_span_a'] < ichimoku_df['future_span_b'], ['future_cloud_over_price']] = ichimoku_df['future_cloud_over_price'] *-1

    # Create new ichimoku columns for use in vertical area charts
    ichimoku_df['span_a_vertical'] = np.where(ichimoku_df['span_a'] < ichimoku_df['span_b'], ichimoku_df['span_a'], ichimoku_df['span_b'])
    ichimoku_df['span_b_vertical'] = np.where(ichimoku_df['span_a'] > ichimoku_df['span_b'], ichimoku_df['span_b'], ichimoku_df['span_a'])
    ichimoku_df['cloud_over_price_vertical'] = np.where((ichimoku_df['cloud_over_price'] > 0) &
                                                        (ichimoku_df['future_cloud_over_price'] < 0),
                                                        ichimoku_df['future_cloud_over_price'],
                                                        ichimoku_df['cloud_over_price']
                                                       )
    ichimoku_df['future_cloud_over_price_vertical'] = np.where((ichimoku_df['cloud_over_price'] < 0) &
                                                               (ichimoku_df['future_cloud_over_price'] > 0),
                                                               ichimoku_df['cloud_over_price'],
                                                               ichimoku_df['future_cloud_over_price']
                                                              )


    for window in windows:
        bbands_df['band_position' + str(window)] = bbands_df['closing_price_band_position'].rolling(window=window, min_periods=window).mean()

    bbands_df = bbands_df.dropna(axis=0)

    ### Make Buy/Sell Prediction - Random Forest Model

    # Load Random Forest Classifier models from pickles
    root = str(Path(".")) +'/random_forest_pickles'

    with open(root + '/rf_classifier_all.pickle', 'rb') as read_file:
        rf_classifier_all = pickle.load(read_file)

    with open(root + '/rf_classifier_all.combined', 'rb') as read_file:
        rf_classifier_combined = pickle.load(read_file)

    with open(root + '/rf_classifier_intuitive.pickle', 'rb') as read_file:
        rf_classifier_intuitive = pickle.load(read_file)

    # Calculate additional fields for Random Forest predictions

    # New addresses vs. active addresses, 30 day average
    glassnode_df = glassnode_df.merge(candles_df[['date', 'close']], how='left', on='date')
    glassnode_df['new_over_active'] = glassnode_df['new_addresses'] / glassnode_df['active_addresses'].shift(1)
    glassnode_df['new_over_active_30'] = glassnode_df['new_over_active'].rolling(window=30, min_periods=30).mean()

    # Number of transactions and transaction fees, 30 day average
    glassnode_df['transactions_30'] = glassnode_df['transactions'].rolling(window=30, min_periods=30).mean()
    glassnode_df['transaction_fees_30'] = glassnode_df['transaction_fees'].rolling(window=30, min_periods=30).mean()

    # Mining difficulty, normalized by price, 30 day average
    glassnode_df['mining_difficulty_norm'] = glassnode_df['mining_difficulty'] / glassnode_df['close']
    glassnode_df['mining_difficulty_norm_30'] = glassnode_df['mining_difficulty_norm'].rolling(window=30, min_periods=30).mean()

    # Add regression column for rsi, 120 days
    rsi_df = add_regression_column(rsi_df, 'rsi', 120)

    # Find Random Forest inputs for the prediction date (yesterday)
    pred_date = rsi_df['date'].iloc[-1]
    glassnode_pred = glassnode_df[glassnode_df['date'] == pred_date]
    ichimoku_pred = ichimoku_df[ichimoku_df['date'] == pred_date]
    rsi_pred = rsi_df[rsi_df['date'] == pred_date]

    stfd = glassnode_pred['stfd'].iloc[0]
    nupl = glassnode_pred['nupl'].iloc[0]
    market_to_thermo = glassnode_pred['market_to_thermo'].iloc[0]
    new_over_active_30 = glassnode_pred['new_over_active_30'].iloc[0]
    transactions_30 = glassnode_pred['transactions_30'].iloc[0]
    transaction_fees_30 = glassnode_pred['transaction_fees_30'].iloc[0]
    mining_difficulty_norm = glassnode_pred['mining_difficulty_norm'].iloc[0]
    t_over_k_streak = ichimoku_pred['t_over_k_streak'].iloc[0]
    price_over_cloud_top = ichimoku_pred['price_over_cloud_top'].iloc[0]
    cloud_over_price = ichimoku_pred['cloud_over_price'].iloc[0]
    rsi_reg_norm120 = rsi_pred['rsi_reg_norm120'].iloc[0] 

    random_forest_dict = {'stfd' : stfd,
                          'nupl' : nupl,
                          'market_to_thermo' : market_to_thermo,
                          'new_over_active_30' : new_over_active_30,
                          'transactions_30' : transactions_30,
                          'transaction_fees_30' : transaction_fees_30,
                          'mining_difficulty_norm' : mining_difficulty_norm,
                          't_over_k_streak' : t_over_k_streak,
                          'price_over_cloud_top' : price_over_cloud_top,
                          'cloud_over_price' : cloud_over_price,
                          'rsi_reg_norm120' : rsi_reg_norm120
                         }

    # Specify the features required for each of the Random Forest models
    intuitive_features = ['stfd', 'new_over_active_30', 'transactions_30', 't_over_k_streak',
                          'market_to_thermo', 'cloud_over_price', 'transaction_fees_30']

    all_features = ['stfd', 'mining_difficulty_norm', 'rsi_reg_norm120', 
                    't_over_k_streak', 'nupl', 'transaction_fees_30']

    combined_features = ['new_over_active_30', 'price_over_cloud_top', 'mining_difficulty_norm', 'stfd', 
                         'market_to_thermo', 'nupl', 'transaction_fees_30']

    def rf_prediction(classifier, features, value_dict):
        """
        Return prediction and confidence level for a given classifier and feature set.
        """

        inputs = []

        for i in features:
            inputs.append(value_dict[i])

        inputs = np.array(inputs).reshape(1,len(inputs))

        pred = classifier.predict(inputs)
        prob = classifier.predict_proba(inputs)

        return pred, prob

    # Make buy/sell predictions
    pred_all, prob_all = rf_prediction(rf_classifier_all, all_features, random_forest_dict)
    pred_intuitive, prob_intuitive = rf_prediction(rf_classifier_intuitive, intuitive_features, random_forest_dict)
    pred_combined, prob_combined = rf_prediction(rf_classifier_combined, combined_features, random_forest_dict) 

    rf_dict = {'all' : [pred_all, prob_all],
               'intuitive' : [pred_intuitive, prob_all],
               'combined' : [pred_all, prob_all]}

    import warnings
    warnings.filterwarnings('ignore')
    
    # Display the doc
    modify_doc(doc = curdoc(), 
               rf_dict=rf_dict,
               ichimoku_df=ichimoku_df, 
               candles_df=candles_df,
               macd_df=macd_df,
               rsi_df=rsi_df,
               bbands_df=bbands_df,
               volume_df=volume_df,
               glassnode_df=glassnode_df,
               divergence_df=divergence_df
              )

main()