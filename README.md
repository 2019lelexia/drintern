# dr_intern
# drintern
`rec.py` 以及 `rec_mono.py` 均为接受rosbag话题同时按去畸变并且转到光轴平行位置的，mono是只有左目的。
`stereo_rectify.py` 是离线检查左右目去畸变以及转到光轴平行位置的效果的。
`align.py` 以及 `alignf.py` 均为可视化检查stereo地图与雷达点云地图重合情况的，其中双目深度点云是FoundationStereo出的。
`bag2folder...` 都是用于从rosbag里得到双目图像并存到文件夹的，因为matlab标定不能用rosbag，而且这样便于直观查看。
