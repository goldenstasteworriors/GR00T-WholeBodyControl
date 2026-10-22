# 1. 一键开启调试夹爪网页

整行复制执行，无需换行。

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/scripts/start_dex1_web.sh
```

网页地址：

```text
http://127.0.0.1:8766
```

同侧扳机选择第一档。
扳机加同侧侧键选择第二档。
只按侧键仍是松开状态。
侧键指手柄侧面Grip。
左右夹爪独立选择档位。
网页可保存多组场景。
保存本身不会移动夹爪。

直接编辑默认配置：

```text
/home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/.runtime/dex1_profiles/default.json
```

每对数值顺序：左、右。
单位rad，范围0至5.4。
角度越大，开口越大。
当前第二档沿用第一档。
下例第二档为演示值。
按实测修改对应数值。

```json
{
  "version": 1,
  "unit": "rad",
  "released_q": [5.4, 5.4],
  "pressed_q": [4.15, 4.15],
  "pressed_grip_q": [4.0, 4.0]
}
```

released_q：松开扳机。
pressed_q：扳机第一档。
pressed_grip_q：同侧组合第二档。
改后重启采集才会加载。
另存其他名称即可增加场景。
例如red_bottle.json。
采集时选择red_bottle。

# 2. 一键开启PICO Service

整行复制执行，无需换行。

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/scripts/thinkpad_pico_services.sh start
```

已有服务会直接复用。
将头显连接到ThinkPad。
开启追踪和Send。
保持头显唤醒。
然后执行数据采集命令。

# 3. 一键数据采集

整行复制执行，无需换行。
只需修改引号内三个参数。
依次为场景、描述、项目名。

```bash
bash /home/data-collection/GR00T-WholeBodyControl-worktrees/thinkpad-wired-pico-onboard/scripts/collect_on_thinkpad.sh "default" "grab the red bottle" "grab_red_bottle"
```

第一项选择夹爪配置场景。
第二项填写任务英文描述。
第三项填写数据集项目前缀。
脚本自动追加日期与时间。
数据保存在ThinkPad的outputs。
同一任务下一条无需重启。

脚本自动关闭夹爪调试网页。
自动准备机器人夹爪服务。
自动更新并重启相机服务。
固定使用头部RGB视频节点。
自动复用或开启PICO服务。
自动检查双爪和三路相机。
随后打开四个采集窗格。
启动会执行原有初始化。
四键只启用身体策略。
上半身关节目标保持不变。
A+X才采样姿态并平滑接入。
接入先保持原命令0.5秒。
接入每步最多变化0.005rad。

左Grip加右A：开始或停止录制。
开始音上升，停止音下降。
左Grip加右B：丢弃本段。
control按d：阻尼急停并保留。
四键急停同样保留程序。
Ctrl+反斜杠：阻尼后退出。

```text
Ctrl+\
```
