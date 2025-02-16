import cv2
from PIL import ImageGrab
import numpy as np
import pytesseract
import pandas as pd

# 全局变量
drawing = False  # 是否正在绘制
roi_coords = []  # 存储 ROI 的坐标
image = None  # 原始图像
image_copy = None  # 用于绘制的图像副本


# 鼠标回调函数
def select_roi(event, x, y, flags, param):
    global drawing, roi_coords, image_copy

    # 左键按下：开始绘制
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_coords = [(x, y)]  # 记录起始点

    # 鼠标移动：动态绘制矩形框
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 重置图像为原始状态
            image_copy = image.copy()
            # 动态绘制矩形框
            cv2.rectangle(image_copy, roi_coords[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", image_copy)

    # 左键释放：结束绘制并保存 ROI
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_coords.append((x, y))  # 记录结束点

        # 提取 ROI
        (x1, y1), (x2, y2) = roi_coords
        roi = image[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]

        # 将 ROI 转换为灰度图像（Tesseract 对灰度图像识别效果更好）
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


        # 使用 pytesseract 识别文字
        text = pytesseract.image_to_string(roi_gray, lang="chi_sim")  # 使用简体中文模型
        print("识别结果：")
        print(text)
        
        # 在 Excel 中查找匹配的内容
        search_in_excel(text.strip())  # 去除前后空格

        # 重置 ROI 坐标
        roi_coords = []
        
# 在 Excel 中查找匹配的内容
def search_in_excel(search_text):
    # 读取 Excel 文件
    excel_file = "D:\Game\JiangHu\古城遗迹(86条).xls"  # 替换为你的 Excel 文件路径
    try:
        df = pd.read_excel(excel_file)
    except Exception as e:
        print(f"读取 Excel 文件失败: {e}")
        return

    # 查找匹配的行
    matches = df.apply(lambda row: row.astype(str).str.contains(search_text, case=False).any(), axis=1)
    matched_rows = df[matches]

    # 输出匹配的结果
    if not matched_rows.empty:
        print(f"在 Excel 中找到匹配的内容：")
        print(matched_rows)
    else:
        print(f"未找到与 '{search_text}' 匹配的内容。")


# 定义截取区域的坐标 (left, top, right, bottom)
bbox = (1900, 400, 2560, 1440)

# 截取屏幕并转换为 OpenCV 格式
screenshot = ImageGrab.grab(bbox=bbox)
image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# 加载图像（或截取屏幕）
if image is None:
    print("图像加载失败，请检查路径！")
    exit()

# 创建图像副本
image_copy = image.copy()

# 创建窗口并绑定鼠标回调函数
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)

# 主循环
while True:
    cv2.imshow("Select ROI", image_copy)
    key = cv2.waitKey(1) & 0xFF

    # 按下 ESC 键退出 (ASCII 码为 27)
    if key == 27:  # ESC 键
        break

# 释放资源
cv2.destroyAllWindows()
