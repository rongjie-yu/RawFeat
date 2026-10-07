# HPatches 合成 Raw 五档评估草案

状态：2026-10-08完成官方资料、数据版本纠正、完整序列验收和评估器收口；本文件冻结评估协议，不修改已发布v1的训练目标、模型、权重或COCO验证协议。完整数据、cache和部署模型保持不变；正式评估入口现为 batch=1 的 `hpatches-evaluate`，报告与可视化在同一次命令中生成。

原下载路径 `/homes/rongjie/datasets/hpatches-release` 有116个目录（57个i_、59个v_），116个ref.png、580个H_ref_*，没有*.ppm。ref.png是宽65的灰度patch竖向堆叠图；例如i_ajuntament/ref.png为65×55445，即853个65×65 patch。它是patch版本，不能用来评价完整检测、自动匹配和单应性链路。该目录保留；后续主评估改用 `/homes/rongjie/datasets/hpatches-sequences-release`。

[HPatches官方说明](https://github.com/hpatches/hpatches-dataset#full-image-sequences)提供独立的完整序列下载：[hpatches-sequences-release.zip](https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip)。已下载1279545572字节，SHA256 `99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670` 与官方Hugging Face LFS登记一致，并解压到上述新目录。116序列（57光照/59视角）的696张RGB完整PPM全部解码通过，580个H_1_*全部为有限可逆3×3矩阵。源图宽512…3088、高380…2056；逐文件尺寸和SHA记录在数据根目录dataset_integrity.json中。原patch版CSV矩阵对应原始场景坐标，不能直接作用于ref.png/e*.png的堆叠坐标。

评估目标是冻结已发布模型，检验COCO训练之后对真实跨照片光照/视角变化及既定合成传感器噪声的泛化。正式名称建议为 `rawfeat.hpatches.synthetic_raw.v1`。HPatches是处理过的图像；通过InvISP和ELD产生的是合成Canon Raw，不能把结果称为真实Raw相机测试或原生HPatches标准分数。

每个完整序列只采用1→2、1→3、1→4、1→5、1→6五对，直接使用对应官方H。116序列预计580对，其中光照285对、视角295对。全部纳入，不根据模型得分、重叠率或困难程度删除序列。读取失败、异常矩阵等数据问题在预检阶段显式报告；不能静默跳过或把子集叫作完整结果。

两张不同照片分别进入现有sRGB→Canon InvISP→sensor RGB→RGGB Bayer→ELD→packed链路。保留实际光照/视角差异；不增加随机裁剪、颜色增强、随机homography，也不把图1warp后代替真实图2…6。教师只可用于可选诊断，不参与主评估选点、匹配或真值生成。Canon标定、full-DN曝光模型、独立噪声流、白平衡和负值保留沿用v1。

首轮采用 SuperPoint HPatches 参考配置的目标尺寸高480、宽640。每张图先按最大轴比例缩放，使缩放后两个轴都不小于目标，再从中心裁出480×640；不做非等比拉伸，也不填充黑边。这样与当前 RawFeat 的4×240×320输入和像素阈值尺度一致，也接近SuperPoint论文的HPatches设置。原始尺寸/不裁剪测试若以后需要，应另立协议。

每张图保存实际尺寸变换T，更新真值：

\[
H'_{1\to j}=T_j H_{1\to j}T_1^{-1}.
\]

T必须与真正resize/crop的像素中心约定一致。若原图宽高为W,H，OpenCV输出尺寸为W',H'，则实际轴缩放为sx=W'/W、sy=H'/H；中心裁剪左/上偏移为cx,cy，像素中心变换为x'=sx*x+(sx−1)/2−cx、y'=sy*y+(sy−1)/2−cy。把这两个轴的实际sx、sy和偏移写入manifest，更新H'=T_j H_1j T_1^{-1}。不要只缩放RGB却沿用原H，或用与实际插值不一致的名义变换。实施前验证H方向、正逆变换、恒等情况、resize前后投影一致性，并检查几组真实匹配可视化。所有报告的1/3/5px均为处理后640×480图像的像素。

主协议沿用当前COCO评估的噪声解释，但只保留两侧处于同一噪声档的对称输入。训练时两侧 ratio 是独立连续采样的；ratio 不相等的样本可能出现，但没有把 `(1,r)` 或 `(r,1)` 作为显式训练层。因此首轮 HPatches 不把这两类有向条件列为正式条件。每个HPatches参考→目标对使用 clean、ratio1，以及对r∈{4,16,64,100}的对称条件 `(r,r)`。其中1表示启用ELD的ratio1，不是clean：

| 档位 | 参考图ratio | 目标图ratio | ELD |
|---|---:|---:|---|
| clean | clean | clean | 关闭 |
| 1 | 1 | 1 | 两侧独立噪声 |
| r,r | r | r | 两侧独立噪声 |

ratio1仍有ELD噪声，不能冒充clean。每个r>1只报告 `(r,r)`，五个含噪档等权平均H-AUC@5，clean单报。每个条件都使用全部580对，因此首轮总量为每模型 `580*(1 clean + 1 ratio1 + 4)=3480` 对；匹配和RANSAC应复用同一图像/ratio特征缓存，不得重复前向。若以后需要研究曝光不对称，再另立诊断协议，不得把它混入本轮正式主分数。

clean↔noisy不是本首轮主条件，若需要必须单列。

噪声seed在manifest中固定，依据全局seed2027、序列、图像编号、ratio和重复编号稳定生成，不使用Python进程随机hash。同一序列图1在五个配对中复用同一图像/ratio的噪声实例与特征；不同图像的噪声独立。所有模型和基线读取同一批实际noisy Bayer数据，不分别随机合成。

首轮冻结比较 `weights/v1/checkpoint_step_096000.pt` 与当前MeanAD+SuperPoint。需要在HPatches同一噪声数据上重新评价基线，不能套用COCO的65.6030%。100000和55000作为预先登记的次级候选；若评估它们，完整报告，不能只展示HPatches上最高的一个再称其为独立测试结果。模型eval、BN固定、不做域适配或更新；不需要重新训练。

所有方法采用相同的处理后尺寸、阈值0.005、NMS4、border8、max1024、MNN和RANSAC（3px、confidence0.999、maxIters10000、固定seed）。保持整数选点及128维学生描述子；不引入局部质心或更换匹配器。基线继续使用现有MeanAD传感器DN处理，不能改用原sRGB作为含噪主基线。官方SuperPoint文献的NMS8/max1000/shared-point数字只作参考，不混入主结果。

检测时只排除真实输入边界，不用GT homography计算共同重叠区去筛除特征。完整照片中非重叠区也是真实可见内容；两侧全图特征参与MNN/RANSAC。GT共同可见区仅用于重复性等评估分母，避免把真值用于改善自动匹配。现有COCO `valid_masks(H)` 包含合成warp的无效填充逻辑，不能直接用于HPatches完整照片。

输出以完整自动链路为中心，按每档的all/illumination/viewpoint分别报告H-AUC@1/3/5、角点误差≤1/3/5px比例、失败数、匹配精度、正确匹配数、点数、重复性和条件定位误差。H-AUC继续使用当前归一化积分定义；另报阈值成功率方便理解和对照，不能把两者混名。记录角点误差中位数、95分位和>20px的对数，失败保留为失败且AUC为0。

all结果按116序列等权；每序列都有五对，这与580对等权一致。illumination和viewpoint单报；不把57/59两类人为再各加权50%而改变all定义。五档主分数仍各加权20%。不确定性按序列进行配对cluster bootstrap（固定seed、10000次），同一序列的五目标和所有噪声条件共同重采样；不能把参考图重复的580对当作完全独立样本。

结果表可采用以下结构，五噪声均值在单独汇总行展示：

| 条件 | 学生all H-AUC@5 | 基线all H-AUC@5 | 差值/区间 | 学生illumination | 学生viewpoint |
|---|---:|---:|---|---:|---:|
| clean | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |
| 1 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |
| 4 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |
| 16 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |
| 64 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |
| 100 | 待评估 | 待评估 | 待评估 | 待评估 | 待评估 |

实现时新增HPatches的manifest/数据读取与评估入口，复用当前sensor/noise/features/baseline/metrics。现有COCO read_manifest硬性规定256源1536对，PairGenerator从同一张图合成两视角，因此都不能直接换一个root路径复用。无需复制第二套模型、训练器、noise实现或历史loss；COCO协议保持独立，两个结果不能合并成一项分数。

缓存以单图和条件为单位，不以图像对重复保存参考图。696张原图分别生成clean base，再生成ratio1/4/16/64/100五个固定噪声版本；只缓存必要的FP32 noisy Bayer和共享元数据，packed按需由当前函数产生。不把可能为负/超白的noisy DN存成uint16，不为节省空间悄悄改成FP16。参考特征也按checkpoint+图像+条件复用；`(r,r)`只改变配对索引，不重复图像前向。源文件、尺寸/T/H、InvISP/ELD/教师/检查点SHA、噪声seed和评估代码/依赖版本均进入身份记录。

推进顺序：

1. 完整序列下载与文件/尺寸/H的CPU预检已完成；实施时以dataset_integrity.json为来源登记，另冻结含预处理/噪声条件的评估manifest。
2. 完成新数据入口和坐标/噪声/掩码契约测试，不更新模型。
3. 在预先固定的2个illumination和2个viewpoint序列做GPU smoke（20基础对、clean+ratio1+4个r的对称条件；最多120 pair/model，实际按缓存前向统计），检查Raw数值域、H方向、匹配可视化、模型/基线数据一致性、失败保留及耗时/显存。
4. 按实测吞吐给出完整3480对/模型的命令和预算，由用户启动耗时评估；先比较96000与基线。第一轮使用1个固定噪声重复并明确此限制。
5. 保存逐对数据、每档/每类型统计与噪声曲线；必要时另外评价已登记的100000/55000候选。如果差异接近噪声随机性，再单独登记3个重复的扩展预算：580×(1+5×3)=9280对/模型，不能自动无限扩张。

现有patch版可保留给未来的描述子已知位置辅助实验；它已经经过patch选取/几何扰动，且为灰度数据，不能替代以上完整链路验证，也不能据此确认检测或Raw域泛化。
