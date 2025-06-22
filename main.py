import os
from uuid import  uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import subprocess
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import util

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

PROCESSED_DIR = os.path.join("static", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

PLOT_DIR = os.path.join(PROCESSED_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

COLOR_DIR = os.path.join("static", "processed", "colors")
os.makedirs(COLOR_DIR, exist_ok=True)

TEXTURE_DIR = os.path.join("static", "processed", "texture")
os.makedirs(TEXTURE_DIR, exist_ok=True)


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("home.html", {"request":request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path
    })

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    #np.full(img.shape, value) = create array of imageshape and the value is value

    #add image value using it's image information(double image)
    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "substract":
        result_img = cv2.substract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max": #maximum operation 
        result_img= np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
        modified_path = save_image(result_img, "modified")
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar", status_code=400)
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        if operation == "and":
            result_img = cv2.bitwise_and(img1, img2)
        elif operation == "xor":
            result_img = cv2.bitwise_xor(img1, img2)

    modified_path = save_image(result_img, img2)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path":  original_path,
        "modified_image_path": modified_path
    })
@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request:Request):
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })



@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})

@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    equalized_img = cv2.equalizeHist(img)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)
		
		#jika ingin grayscale
    #img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    #ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR


    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage #grayscale
    #specified_img = match_histograms(img, ref_img, multichannel=False)
		    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 5)

@app.get("/capture_faces", response_class=HTMLResponse)
async def capture_faces_form(request: Request):
    return templates.TemplateResponse("capture_faces.html", {"request": request})

@app.post("/capture_faces", response_class=HTMLResponse)
async def capture_faces(request: Request, name: str = Form(...)):
    if not name.strip():
        return templates.TemplateResponse("capture_faces.html", {
            "request": request,
            "message": "Nama tidak boleh kosong.",
            "success": False
        })

    save_path = os.path.join('dataset', name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return templates.TemplateResponse("capture_faces.html", {
            "request": request,
            "message": "Tidak dapat mengakses webcam.",
            "success": False
        })

    num_images = 0
    max_images = 20

    try:
        while num_images < max_images:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                img_name = os.path.join(save_path, f"img_{num_images}.jpg")
                cv2.imwrite(img_name, face_img)
                num_images += 1
                break

            time.sleep(0.1)
    finally:
        cap.release()

    return templates.TemplateResponse("capture_faces.html", {
        "request": request,
        "message": f"{num_images} gambar berhasil disimpan untuk {name}.",
        "success": True
    })

def generate_freeman_chain_code(contour):
    directions = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
    }

    chain_code = []
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        norm_dx, norm_dy = np.sign(dx), np.sign(dy)
        code = directions.get((norm_dx, norm_dy))
        if code is not None:
            chain_code.append(code)
    return chain_code


@app.get("/chain_codes", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chain_code.html", {"request": request})


@app.post("/chain_codes", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    img = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    chain_code_text = "Tidak ada kontur ditemukan."
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [largest], -1, (0, 255, 0), 1)
        chain_code = generate_freeman_chain_code(largest)

        wrapped = ""
        line_len = 70
        current = 0
        for i, c in enumerate(map(str, chain_code)):
            item = c + (", " if i < len(chain_code) - 1 else "")
            if current + len(item) > line_len:
                wrapped += "\n"
                current = 0
            wrapped += item
            current += len(item)

        chain_code_text = (
            f"Jumlah Kontur: {len(contours)}\n"
            f"Panjang Kode: {len(chain_code)}\n"
            f"Kode Rantai:\n{wrapped}"
        )

    # Save processed images for display
    vis_path = os.path.join("static", "processed")
    os.makedirs(vis_path, exist_ok=True)
    img_name = f"{file_id}_gray.jpg"
    bin_name = f"{file_id}_binary.jpg"
    cont_name = f"{file_id}_contour.jpg"

    cv2.imwrite(os.path.join(vis_path, img_name), img)
    cv2.imwrite(os.path.join(vis_path, bin_name), binary)
    cv2.imwrite(os.path.join(vis_path, cont_name), contour_img)

    return templates.TemplateResponse("chain_code.html", {
        "request": request,
        "gray_img": f"/static/processed/{img_name}",
        "binary_img": f"/static/processed/{bin_name}",
        "contour_img": f"/static/processed/{cont_name}",
        "chain_code_text": chain_code_text
    })

@app.get("/canny", response_class=HTMLResponse)
async def show_canny_form(request: Request):
    return templates.TemplateResponse("canny_edges.html", {"request": request})


@app.post("/canny", response_class=HTMLResponse)
async def process_canny_edge(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid4())
    file_ext = os.path.splitext(file.filename)[-1]
    raw_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    img = cv2.imread(raw_path)
    if img is None:
        return templates.TemplateResponse("canny_edges.html", {
            "request": request,
            "error": f"Gagal membaca citra dari {file.filename}"
        })

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    low_threshold, high_threshold = 50, 150
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    orig_path = os.path.join(PROCESSED_DIR, f"{file_id}_orig.jpg")
    blur_path = os.path.join(PROCESSED_DIR, f"{file_id}_blur.jpg")
    edge_path = os.path.join(PROCESSED_DIR, f"{file_id}_edge.jpg")

    cv2.imwrite(orig_path, img)
    cv2.imwrite(blur_path, blurred)
    cv2.imwrite(edge_path, edges)

    return templates.TemplateResponse("canny_edges.html", {
        "request": request,
        "orig_img": f"/static/processed/{os.path.basename(orig_path)}",
        "blur_img": f"/static/processed/{os.path.basename(blur_path)}",
        "edge_img": f"/static/processed/{os.path.basename(edge_path)}",
        "low_threshold": low_threshold,
        "high_threshold": high_threshold
    })

@app.get("/integral_projection", response_class=HTMLResponse)
async def show_projection_form(request: Request):
    return templates.TemplateResponse("projection_analysis.html", {"request": request})


@app.post("/integral_projection", response_class=HTMLResponse)
async def process_projection(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid4())
    file_ext = os.path.splitext(file.filename)[-1]
    raw_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return templates.TemplateResponse("projection_analysis.html", {
            "request": request,
            "error": f"Gagal membaca citra dari {file.filename}"
        })

    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_norm = binary_img / 255.0

    horizontal_projection = np.sum(binary_norm, axis=0)
    vertical_projection = np.sum(binary_norm, axis=1)

    height, width = binary_norm.shape

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.imshow(binary_norm, cmap='gray')
    ax_img.set_title('Citra Biner (Objek=1)')
    ax_img.set_xlabel('Indeks Kolom')
    ax_img.set_ylabel('Indeks Baris')

    ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
    ax_hproj.plot(np.arange(width), horizontal_projection)
    ax_hproj.set_title('Proyeksi Horizontal')
    ax_hproj.set_ylabel('Jumlah Piksel')
    plt.setp(ax_hproj.get_xticklabels(), visible=False)

    ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
    ax_vproj.plot(vertical_projection, np.arange(height))
    ax_vproj.set_title('Proyeksi Vertikal')
    ax_vproj.set_xlabel('Jumlah Piksel')
    ax_vproj.invert_yaxis()
    plt.setp(ax_vproj.get_yticklabels(), visible=False)

    plt.suptitle("Analisis Proyeksi Integral", fontsize=14)
    
    plot_filename = f"{file_id}_projection.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)

    return templates.TemplateResponse("projection_analysis.html", {
        "request": request,
        "plot_img": f"/static/processed/plots/{plot_filename}"
    })

@app.get("/compression", response_class=HTMLResponse)
async def show_compression_form(request: Request):
    return templates.TemplateResponse("compression_analysis.html", {"request": request})


@app.post("/compression", response_class=HTMLResponse)
async def process_compression(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid4())
    ext = os.path.splitext(file.filename)[-1].lower()
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        return templates.TemplateResponse("compression_analysis.html", {
            "request": request,
            "error": f"Gagal membaca citra dari {file.filename}"
        })

    is_color = len(img_bgr.shape) == 3
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if is_color else img_bgr
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if is_color else img_bgr
    original_cv = img_rgb if is_color else img_gray

    original_size = os.path.getsize(input_path)

    min_dim = min(original_cv.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(win_size, 3)

    results = []
    jpeg_qualities = [95, 75, 50, 25, 10]
    for q in jpeg_qualities:
        out_jpeg = os.path.join(PROCESSED_DIR, f"{file_id}_q{q}.jpg")
        save_img = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR) if is_color else original_cv
        cv2.imwrite(out_jpeg, save_img, [cv2.IMWRITE_JPEG_QUALITY, q])

        comp_img = cv2.imread(out_jpeg)
        comp_cv = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB) if is_color else comp_img

        psnr_val = cv2.PSNR(original_cv, comp_cv)
        ssim_val = ssim(original_cv, comp_cv, channel_axis=2 if is_color else None, win_size=win_size,
                        data_range=original_cv.max() - original_cv.min())

        results.append({
            "Method": "JPEG",
            "Quality": q,
            "SizeKB": os.path.getsize(out_jpeg) / 1024,
            "Ratio": original_size / os.path.getsize(out_jpeg),
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })

    png_levels = range(10)
    for l in png_levels:
        out_png = os.path.join(PROCESSED_DIR, f"{file_id}_l{l}.png")
        save_img = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR) if is_color else original_cv
        cv2.imwrite(out_png, save_img, [cv2.IMWRITE_PNG_COMPRESSION, l])

        comp_img = cv2.imread(out_png)
        comp_cv = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB) if is_color else comp_img

        psnr_val = cv2.PSNR(original_cv, comp_cv)
        try:
            ssim_val = ssim(original_cv, comp_cv, channel_axis=2 if is_color else None, win_size=win_size,
                            data_range=original_cv.max() - original_cv.min())
        except ValueError:
            ssim_val = None

        results.append({
            "Method": f"PNG Level {l}",
            "Quality": "Lossless",
            "SizeKB": os.path.getsize(out_png) / 1024,
            "Ratio": original_size / os.path.getsize(out_png),
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })

    df = pd.DataFrame(results)
    plot_file = os.path.join(PLOT_DIR, f"{file_id}_compression_plot.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x="Method", y=["PSNR", "SSIM"], kind="bar", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    size_plot_file = os.path.join(PLOT_DIR, f"{file_id}_size_compression_plot.png")
    df.plot(x="Method", y=["SizeKB"], kind="bar", ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(size_plot_file)
    plt.close()


    return templates.TemplateResponse("compression_analysis.html", {
        "request": request,
        "plot_img": f"/static/processed/plots/{os.path.basename(plot_file)}",
        "size_plot_img": f"/static/processed/plots/{os.path.basename(size_plot_file)}",
        "results": df.to_dict(orient="records")
    })


def save_image_plot(image, title, cmap=None):
    filename = f"{uuid4().hex}.png"
    path = os.path.join(COLOR_DIR, filename)
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return f"/static/processed/colors/{filename}"


def save_channels_plot(channels, titles, cmap='gray'):
    n = len(channels)
    filename = f"{uuid4().hex}.png"
    path = os.path.join(COLOR_DIR, filename)

    plt.figure(figsize=(4*n, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(channels[i], cmap=cmap)
        plt.title(titles[i])
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return f"/static/processed/colors/{filename}"


def rgb_to_yiq(rgb):
    rgb_norm = rgb.astype(np.float32) / 255.0
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    h, w, _ = rgb_norm.shape
    reshaped = rgb_norm.reshape(h * w, 3)
    yiq = np.dot(reshaped, matrix.T).reshape(h, w, 3)
    return yiq


@app.get("/color", response_class=HTMLResponse)
async def show_color_analysis_form(request: Request):
    return templates.TemplateResponse("color_analysis.html", {"request": request})


@app.post("/color", response_class=HTMLResponse)
async def analyze_color_spaces(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid4())
    ext = os.path.splitext(file.filename)[-1].lower()
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image = cv2.imread(input_path)
    if image is None:
        return templates.TemplateResponse("color_analysis.html", {
            "request": request,
            "error": "Gagal membaca citra."
        })

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_img_path = save_image_plot(image_rgb, "Citra RGB Asli")

    R, G, B = cv2.split(image_rgb)
    rgb_channels_path = save_channels_plot([R, G, B], ['Red', 'Green', 'Blue'])

    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    lab_img_path = save_image_plot(image_lab, "Citra dalam Lab")
    L, a, b = cv2.split(image_lab)
    lab_channels_path = save_channels_plot([L, a, b], ['L', 'a', 'b'])

    image_ycbcr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    ycbcr_img_path = save_image_plot(image_ycbcr, "Citra dalam YCbCr")
    Y, Cr, Cb = cv2.split(image_ycbcr)
    ycbcr_channels_path = save_channels_plot([Y, Cb, Cr], ['Y', 'Cb', 'Cr'])

    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hsv_img_path = save_image_plot(image_hsv, "Citra dalam HSV")
    H, S, V = cv2.split(image_hsv)
    hsv_channels_path = save_channels_plot([H, S, V], ['H', 'S', 'V'])

    image_yiq = rgb_to_yiq(image_rgb)
    yiq_img_path = save_image_plot(image_yiq, "Citra dalam YIQ")
    Yi, Ii, Qi = image_yiq[:, :, 0], image_yiq[:, :, 1], image_yiq[:, :, 2]
    yiq_channels_path = save_channels_plot([Yi, Ii, Qi], ['Y', 'I', 'Q'])

    luminance_components = {
    'Y dari YCbCr': Y,
    'L dari Lab': L,
    'Y dari YIQ': Yi * 255,  # Skalakan kembali ke rentang 0-255
    'V dari HSV': V
    }

    plt.figure(figsize=(12, 8))
    i = 1
    for name, component in luminance_components.items():
        plt.subplot(2, 2, i)
        plt.imshow(component, cmap='gray')
        plt.title(name)
        plt.axis('off')
        i += 1
    plt.tight_layout()
    final_luminance_plot = os.path.join(PLOT_DIR, f"{file_id}_luminance_compare.png")
    plt.savefig(final_luminance_plot)
    plt.close()

    return templates.TemplateResponse("color_analysis.html", {
        "request": request,
        "result": {
            "original": rgb_img_path,
            "rgb_channels": rgb_channels_path,
            "lab": lab_img_path,
            "lab_channels": lab_channels_path,
            "ycbcr": ycbcr_img_path,
            "ycbcr_channels": ycbcr_channels_path,
            "hsv": hsv_img_path,
            "hsv_channels": hsv_channels_path,
            "yiq": yiq_img_path,
            "yiq_channels": yiq_channels_path,
            "luminance_compare": f"/static/processed/plots/{file_id}_luminance_compare.png",
        }
    })

@app.get("/texture", response_class=HTMLResponse)
async def texture_form(request: Request):
    return templates.TemplateResponse("texture_analysis.html", {"request": request})


@app.post("/texture", response_class=HTMLResponse)
async def texture_process(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid4())
    ext = os.path.splitext(file.filename)[-1].lower()
    raw_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    img_bgr = cv2.imread(raw_path)
    if img_bgr is None:
        return templates.TemplateResponse("texture_analysis.html", {
            "request": request,
            "error": f"Gagal membaca citra dari {file.filename}"
        })

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Simpan original + grayscale
    original_combined = np.hstack((img_rgb, cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)))
    original_path = os.path.join(TEXTURE_DIR, f"{file_id}_original.png")
    plt.imsave(original_path, original_combined)

    # GLCM
    distances = [1]
    angles = [0]
    glcm = graycomatrix(img_gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    glcm_fig_path = os.path.join(TEXTURE_DIR, f"{file_id}_glcm.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(glcm[:, :, 0, 0], cmap='viridis')
    plt.title("Matriks GLCM (jarak=1, sudut=0)")
    plt.colorbar(label='Frekuensi')
    plt.tight_layout()
    plt.savefig(glcm_fig_path)
    plt.close()

    # LBP
    lbp = local_binary_pattern(img_gray, 24, 3, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    lbp_path = os.path.join(TEXTURE_DIR, f"{file_id}_lbp.png")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(lbp, cmap='jet')
    plt.title('Peta LBP')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(hist)), hist)
    plt.title('Histogram LBP')
    plt.tight_layout()
    plt.savefig(lbp_path)
    plt.close()

    # Law's
    def laws_energy(image):
        L5 = np.array([1, 4, 6, 4, 1])
        E5 = np.array([-1, -2, 0, 2, 1])
        filters = {
            "L5E5": np.outer(L5, E5),
            "E5S5": np.outer(E5, np.array([-1, 0, 2, 0, -1])),
        }
        results = []
        titles = []
        for name, kernel in filters.items():
            filtered = cv2.filter2D(np.float32(image), -1, kernel)
            energy = cv2.boxFilter(np.abs(filtered), -1, (15, 15))
            norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            results.append(norm)
            titles.append(name)
        return results, titles

    law_imgs, law_titles = laws_energy(img_gray)
    law_path = os.path.join(TEXTURE_DIR, f"{file_id}_law.png")
    plt.figure(figsize=(10, 4))
    for i in range(len(law_imgs)):
        plt.subplot(1, len(law_imgs), i+1)
        plt.imshow(law_imgs[i], cmap='jet')
        plt.title(law_titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(law_path)
    plt.close()

    return templates.TemplateResponse("texture_analysis.html", {
        "request": request,
        "result": {
            "original": f"/static/processed/texture/{os.path.basename(original_path)}",
            "glcm": f"/static/processed/texture/{os.path.basename(glcm_fig_path)}",
            "lbp": f"/static/processed/texture/{os.path.basename(lbp_path)}",
            "law": f"/static/processed/texture/{os.path.basename(law_path)}"
        }
    })



