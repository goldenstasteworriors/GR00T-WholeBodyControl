# 流程条件下的初始目标框

当前状态：离线人工框已保存；本文记录后续 Qwen 视觉定位的输入输出约定，尚未接入在线模型服务。

## 目标身份先于坐标

流程步骤必须提供基本任务、目标物／支撑物和具体目标位置。左／中／右应由流程指定并定义参照物、观察方向、候选数量和排序方式；不能由模型任意选择显眼物体，也不能使用固定像素位置代替目标语义。本次历史数据修正只依照实际放回过程，不补造左／中／右标签。

放回试管时，目标是流程指定的放置位置。初始框表示试管放回后占据的完整图像区域；目标可以为空，不能强制选取另一支已经存在的试管。框的表示为原始 ego 图像像素 xyxy，右下边界不包含在框内。部分出画、遮挡、目标歧义必须明确区分，不把邻管当成缺失目标。

## 在线与离线证据

在线输入：当前 ego 图像、当前流程步骤、指定位置与场景参照；必要时使用当前时刻之前的场景参考帧和已知取管位置。在线定位不能读取任务完成后的未来帧。每个基本操作开始时定位一次，并将图像时间戳、流程条件、框、目标身份和定位状态一并记录；坐标和用于推理的图像必须对应同一帧。

离线复核可以参考放回前、松爪附近、放回后的 ego 和腕部图像，辨别真正新增的试管。背景配准只用于相机移动后的坐标对应，不能证明被选物体就是正确目标。

## 英文定位指令草案

```text
Ground the destination specified by the current procedure step in the current ego image.
Use the procedure's target description and reference frame to identify the intended destination before predicting coordinates.
For RETURN_TUBE_TO_RACK, return the full image region that the returned tube would occupy at the designated slot, not only the small aperture and not an unrelated occupied tube.
An empty destination is valid. Do not select a neighboring visible tube merely because the intended destination is empty or occluded.
Use only the current image and explicitly provided past context. Never use future completion images for online inference.
Return: task, target_description, target_identity, grounding_status, bbox_xyxy, image_width, image_height, image_timestamp, and evidence.
Use integer coordinates in the original image. The maximum corner is exclusive.
If the destination cannot be identified uniquely, set grounding_status to ambiguous or not_visible and bbox_xyxy to null.
If only part is visible, set grounding_status to partial and clip the box to the image boundary.
```

尚待在线接入时确定：实际 Qwen 视觉模型及服务地址、调用环境、流程字段来源与场景参照定义。现有人工框和复核图可用作示范与离线评测，不应直接视作已经验证的在线性能。
