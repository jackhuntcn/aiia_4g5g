# aiia_4g5g

比赛链接：http://36.133.53.121:1080/index/content_page?pageType=11#introduce


## 依赖软件

- fbprophet
- catboost
- xgboost

## 复现步骤：

- 01_extrace_data.py       提取时间序列数据(只使用六月份的数据)
- 01_1_save_maxmin.py      提取时间序列中的每个小区的每个指标的最大最小值, 后面会用来做截断
- 02_rule_submit.py        模型1: 规则模型, 使用每个小区的最后一天(2021/06/30)的数据来循环填充未来七天
- 03_prophet_submit.py     模型2: Prophet, 目标 boxcox 转换
- 04_create_features.py    生成后面的树模型需要用到的特征
- 05_catboost_submit.py    模型3: catboost 树模型, 目标 log1p 转换
- 06_xgboost_submit.py     模型4: xgboost 树模型, 目标 log1p 转换, 与 catboost 采用同一套特征
- 07_blend.py              融合四个模型(0.25:0.40:0.25:0.10)生成最终的提交结果

## 提交结果

- rule: 0.3490
- prophet: 0.3460
- catboost: 0.3106
- xgboost: 0.3146
- blend: 0.2848

## 特征工程

- opt 小区信息基础特征
- opt 目标编码统计特征 (掉分)
- series user 目标编码
- series user_day 目标编码
- series user_hour 目标编码 (上分较多)
- series user_hour 变化量统计特征
- series user_weekday 变化量编码 (少量掉分)
- series user_hour sliding_windows 目标编码 (掉分)
- ......

## 后处理

- 流量类指标截断到小区历史的 0.8x最小值，1.2x最大值
- 其他指标截断到小区历史的 1.0x最小值, 1.0x最大值

## 其他无效尝试

- 指数平滑模型 statsmodels.tsa.holtwinters.ExponentialSmoothing 只有 0.64
- MLP，采用 catboost 那套特征, 无法收敛 (已 minmaxscaler)
- arima 速度慢，auto arima 更不能接受
- neural_prophet 九天环境无法安装
- lightgbm 使用 CPU 速度太慢, 不好迭代
- 使用五六两个月份跑树模型, 掉分明显
- ......
