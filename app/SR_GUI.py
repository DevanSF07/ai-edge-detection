import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None
from utils import utils_image as util
import torch
import threading
import queue
from models.network import RRDBNet as net
import time

def super_resolve(image, model, device):
    img_L = util.uint2tensor4(image)
    img_L = img_L.to(device)
    img_E = model(img_L)
    img_E = util.tensor2uint(img_E)
    return img_E

def init_model(device):
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)  # define network
    model_path = os.path.join('model_zoo', 'Model.pth')
    model.load_state_dict(torch.load(model_path), strict=True)
    model._save_to_state_dict = True  # Enable saving to state_dict   
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model

class SuperResolutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SuperResolution Program")

        self.image_path = ""
        self.save_path = ""
        self.region_file=''
        self.image = None
        self.image_tk = None
        self.canvas = None
        self.rect = None  # 用于存储绘制的矩形框 Used to store the drawn rectangle frame.

        self.progress_var = tk.DoubleVar()
        self.queue = queue.Queue()

        self.mode_1_frame = None
        self.mode_2_frame = None
        self.create_widgets()

    def create_widgets(self):
        # 模式选择按钮 Mode selection button
        self.mode_frame = tk.Frame(self.root)
        self.mode_frame.grid(row=0, column=0, padx=10, pady=10)

        self.mode1_button = tk.Button(self.mode_frame, text="Mode 1: Whole Image", command=self.switch_mode_1)
        self.mode1_button.grid(row=0, column=0, padx=10, pady=10)

        self.mode2_button = tk.Button(self.mode_frame, text="Mode 2: Manual Select", command=self.switch_mode_2)
        self.mode2_button.grid(row=1, column=0, padx=10, pady=10)

    def switch_mode_1(self):
        self.clear_frame()  # 清除当前界面 Clear the current interface
        self.mode_1_frame = tk.Frame(self.root)
        self.mode_1_frame.grid(row=0, column=1, padx=10, pady=10)

        # 修改模式按钮的样式 Modify the style of the mode button
        self.mode1_button.config(bg="lightblue", state="normal")  # 高亮当前模式按钮并禁用 Highlight the current mode button and disable it.
        self.mode2_button.config(bg="lightgray", state="normal")  # 禁用另一个模式按钮 Disable the other mode button

        # 模式一界面组件
        self.select_image_button = tk.Button(self.mode_1_frame, text="Select image", command=self.select_image)
        self.select_image_button.grid(row=0, column=0, padx=10, pady=10)

        self.select_save_button = tk.Button(self.mode_1_frame, text="Select save path", command=self.select_save)
        self.select_save_button.grid(row=1, column=0, padx=10, pady=10)

        self.image_path_label = tk.Label(self.mode_1_frame, text="Image path: Not selected")
        self.image_path_label.grid(row=0, column=1, padx=10, pady=10)

        self.save_path_label = tk.Label(self.mode_1_frame, text="Save path: Not selected")
        self.save_path_label.grid(row=1, column=1, padx=10, pady=10)

        self.progress_bar = ttk.Progressbar(self.mode_1_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.grid(row=2, column=0, padx=10, pady=20)

        self.start_button = tk.Button(self.mode_1_frame, text="Start processing", command=self.start_processing_mode_1)
        self.start_button.grid(row=3, column=0, padx=10, pady=10)

    def switch_mode_2(self):
        self.clear_frame()  # Clear the current interface
        self.mode_2_frame = tk.Frame(self.root)
        self.mode_2_frame.grid(row=0, column=1, padx=10, pady=10)

        # 修改模式按钮的样式 Modify the style of the mode button
        self.mode1_button.config(bg="lightgray", state="normal")  # 高亮当前模式按钮并禁用 Highlight the current mode button and disable it.
        self.mode2_button.config(bg="lightblue", state="normal")  # 禁用另一个模式按钮 Disable the other mode button
        # 模式二界面组件
        self.select_image_button = tk.Button(self.mode_2_frame, text="Select image", command=self.select_image)
        self.select_image_button.grid(row=0, column=0, padx=10, pady=10)

        self.select_save_button = tk.Button(self.mode_2_frame, text="Select save path", command=self.select_save)
        self.select_save_button.grid(row=1, column=0, padx=10, pady=10)

        self.image_path_label = tk.Label(self.mode_2_frame, text="Image path: Not selected")
        self.image_path_label.grid(row=0, column=1, padx=10, pady=10)

        self.save_path_label = tk.Label(self.mode_2_frame, text="Save path: Not selected")
        self.save_path_label.grid(row=1, column=1, padx=10, pady=10)

        self.canvas = tk.Canvas(self.mode_2_frame, width=800, height=800)
        self.canvas.grid(row=2, column=0, padx=10, pady=10)

        self.start_button = tk.Button(self.mode_2_frame, text="Start processing", command=self.start_processing_mode_2)
        self.start_button.grid(row=3, column=0, padx=10, pady=10)

    def clear_frame(self):
        """Clear the current interface"""
        if self.mode_1_frame:
            self.mode_1_frame.grid_forget()
        if self.mode_2_frame:
            self.mode_2_frame.grid_forget()

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select an image file",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if file_path:
            self.image_path = file_path
            if self.mode_1_frame:
                self.image_path_label.config(text=f"Image path: {self.image_path}")
            if self.mode_2_frame:
                self.image_path_label.config(text=f"Image path: {self.image_path}")
                self.display_image_on_canvas(file_path)

    def select_save(self):
        folder_path = filedialog.askdirectory(title="Select save path")
        if folder_path:
            self.save_path = folder_path
            if self.mode_1_frame:
                self.save_path_label.config(text=f"Save Path: {self.save_path}")
            if self.mode_2_frame:
                self.save_path_label.config(text=f"Save path: {self.save_path}")

    def display_image_on_canvas(self, image_path):
        # 读取原图 Read the original image
        self.image = Image.open(image_path)

        # 等比例缩小图像 Scale the image proportionally
        max_size = 800  # 缩小后的最大尺寸 Maximum size after reduction（例如 For example1000px）
        width, height = self.image.size
        if width > max_size or height > max_size:
            scaling_factor = min(max_size / width, max_size / height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            self.image_resized = self.image.resize((new_width, new_height))#, Image..ANTIALIAS)
        else:
            self.image_resized = self.image
            scaling_factor = 1.0

        self.image_tk = ImageTk.PhotoImage(self.image_resized)

        if self.canvas:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))  # 更新画布可滚动区域 Update the canvas scrollable area
            self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2)
            self.canvas.bind("<ButtonPress-1>", self.on_press)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # 记录缩放因子，方便计算恢复坐标时使用 Record the zoom factor for convenient use when calculating restored coordinates.

        self.scaling_factor = scaling_factor

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        self.end_x = event.x
        self.end_y = event.y

        # 将矩形区域还原为原始图像的坐标 Restore the rectangular area to the coordinates of the original image.
        if hasattr(self, 'scaling_factor') and self.scaling_factor:
            original_x1 = int(self.start_x / self.scaling_factor)
            original_y1 = int(self.start_y / self.scaling_factor)
            original_x2 = int(self.end_x / self.scaling_factor)
            original_y2 = int(self.end_y / self.scaling_factor)

            # 截取区域
            region = self.image.crop((original_x1, original_y1, original_x2, original_y2))

            # 使用截取坐标作为文件名
            (imgname, imgext) = os.path.splitext(os.path.basename(self.image_path))
            save_filename = f"{imgname}_Cropped_{original_x1}_{original_y1}_{original_x2}_{original_y2}.png"
            self.region_file=os.path.join(self.save_path, save_filename)
            region.save(self.region_file)

            messagebox.showinfo("Completed", f"Screenshot saved to：{self.region_file}")

    def update_progress(self):
        try:
            progress = self.queue.get_nowait()
            self.progress_var.set(progress)
            self.root.update_idletasks()
        except queue.Empty:
            pass

        # 如果处理还没结束，继续更新进度条
        if self.queue.qsize() > 0 or any(thread.is_alive() for thread in threading.enumerate()):
            self.root.after(50, self.update_progress)

    def start_processing_mode_1(self):
        if not self.image_path or not self.save_path:
            messagebox.showerror("Error", "Please select the image and the save path first.！")
            return
        start_time = time.time()

        image = util.imread_uint(self.image_path)
        patches_dir = os.path.join(self.save_path, "Patches")
        os.makedirs(patches_dir, exist_ok=True)

        img_height, img_width, _ = image.shape
        num_patches_x = (img_width // 256)
        num_patches_y = (img_height // 256)
        num_patches = num_patches_x * num_patches_y
        current_patch = 0

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = init_model(device)

        #Save the model in 3 different formats
        # torch.save(model, 'D://Model.pt')  # Save the model state_dict - Saves but cannot read in C++ project
        # # Saving model as .pb file is not supported by PyTorch directly, so we use ONNX format in between
        # # 设置输入张量名，多个输入就是多个名 Set the input tensor names; multiple inputs mean multiple names.
        # model = torch.nn.DataParallel(model)
        # # 设置模型输入维度 Set the model input dimensions
        # input = torch.randn([1,2,3])
        # input_names = ["input"]
        # # 设置输出张量名 Set output tensor name
        # output_names = ["output"]
        # # 自定义onnx文件名和路径 Custom ONNX file name and path
        # onnx_filename = "model.onnx"
        # # 执行转化和保存 Execute transformation and save as onnx file
        # torch.onnx.export(model.module, input, onnx_filename, verbose=True, input_names=input_names, output_names=output_names)
        # # Read ONNX file and save as .pb file
        # onnx_model = torch.onnx.load("model.onnx")  # load onnx model
        # tf_exp = torch.onnx.prepare(onnx_model)  # prepare tf representation
        # tf_exp.export_graph("model.pb")  # export the model
        #

        torch.cuda.empty_cache()

        patches = []  # 用于保存子图的位置和路径 Used to save the position and path of the subgraph

        # 启动线程进行处理 Start the thread for processing
        thread = threading.Thread(target=self.process_patches, args=(image, patches_dir, device, model, num_patches, current_patch, patches))
        thread.start()

        # 定期更新进度条 Regularly update the progress bar
        self.root.after(50, self.update_progress)

    def process_patches(self, image, patches_dir, device, model, num_patches, current_patch, patches):
        img_height, img_width, _ = image.shape
        for y in range(0, img_height, 256):
            for x in range(0, img_width, 256):
                sub_image = image[y:y + 256, x:x + 256]
                sub_image_sr = super_resolve(sub_image, model, device)
                sub_image_sr=cv2.cvtColor(sub_image_sr,cv2.COLOR_BGR2GRAY)
                sub_image_name = f"patch_{current_patch}.png"
                util.imsave(sub_image_sr, os.path.join(patches_dir, sub_image_name))
                print(f"{os.path.join(patches_dir, sub_image_name)}Saved")

                patches.append((x, y, os.path.join(patches_dir, sub_image_name)))  # 保存子图的位置信息 Save the position information of the sub-figure
                current_patch += 1
                progress = (current_patch / num_patches) * 100
                self.queue.put(progress)
                torch.cuda.empty_cache()

        # 合并所有子图 Merge all subplots
        scale=4
        self.merge_patches(scale,scale*(x+256), scale*(y+256), patches)
        
        # messagebox.showinfo("完成", "超分辨率处理完成！") (Completed, Super-resolution processing is completed!)
        self.queue.put(100)  # 处理完毕 Processing completed

        # end_time = time.now()
        # Calculate the execution time
        #elapsed_time = end_time - start_time
        #print('Execution time:', elapsed_time)

    def merge_patches(self, scale, img_width, img_height, patches):
        full_image = Image.new('RGB', (img_width, img_height))

        for (x, y, patch_path) in patches:
            patch = Image.open(patch_path)
            full_image.paste(patch, (scale*x, scale*y))  # 将每个子图按照记录的位置合并 Merge each subgraph according to the recorded positions.

        (imgname, imgext) = os.path.splitext(os.path.basename(self.image_path))
        full_image_path = os.path.join(self.save_path, 'SR_'+imgname+imgext)
        full_image.save(full_image_path,format='PNG', compress_level=0)
        messagebox.showinfo("Completed", f"Super-resolution result saved to：{full_image_path}！")
        print(f"Super-resolution result saved.: {full_image_path}")

    def start_processing_mode_2(self):
        if not self.image_path or not self.save_path:
            messagebox.showerror("Error", "Please select the image and the save path first.！")
            return

        if self.rect:
            # 读取截取区域的图片 Read the image of the captured area
            save_filename = self.region_file
            # 直接修改文件名 Directly modify the file name，将 will 'Cropped' 替换为 'SR' Replace with 'SR
            save_sr_filename = self.image_path.replace(".png", "_SR.png")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = init_model(device)
            torch.cuda.empty_cache()

            # 直接读取保存的图片进行超分辨率处理 Directly read the saved images for super-resolution processing.
            try:
                print(f"Processing {save_filename}...")
                region_to_process = util.imread_uint(self.image_path)
                region_sr = super_resolve(region_to_process, model, device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()  # 清理缓存 Clear cache
                    messagebox.showerror("Insufficient video memory", "Insufficient video memory, please reduce the area of the captured image.。")
                else:
                    messagebox.showerror("Error", f"An error occurred: {e}")
                return

            # 保存超分辨率处理后的图片，文件名将 'Cropped' 改为 'SR' Save the image after super-resolution processing, changing 'Cropped' in the filename to 'SR'.
            util.imsave(region_sr, save_sr_filename)
            messagebox.showinfo("Completed", f"Super-resolution result saved to：{save_sr_filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SuperResolutionApp(root)
    root.mainloop()