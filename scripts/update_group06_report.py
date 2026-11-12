"""Create a truthful, implementation-aligned revision of the supplied group report."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\Admin\Downloads\Nhóm 06_Luân Quốc Cường_CNTT K22A.docx")
OUTPUT = ROOT / "reports" / "Nhom_06_Phat_hien_benh_la_ca_chua_cap_nhat.docx"
ASSETS = ROOT / "reports" / "docx_qa" / "group06_assets"


def make_figures() -> tuple[Path, Path, Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    flow = ASSETS / "system_flow.png"
    fig, ax = plt.subplots(figsize=(11, 3.1))
    ax.axis("off")
    labels = [
        (0.04, "Ảnh tải lên\nhoặc webcam", "#DDEBF7"),
        (0.27, "Tomato gate\naccepted / rejected / unverified", "#FFF2CC"),
        (0.52, "FastAPI + YOLO11s\nlocal", "#E2F0D9"),
        (0.75, "Box + nhãn +\nhướng xử lý", "#FCE4D6"),
    ]
    for x, text, color in labels:
        ax.text(x, 0.52, text, ha="center", va="center", fontsize=10, weight="bold",
                bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "#4F4F4F"})
    for x in (0.155, 0.395, 0.65):
        ax.annotate("", xy=(x + 0.075, 0.52), xytext=(x, 0.52), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(0.5, 0.12, "Mọi suy luận chạy local. Ảnh bị gate từ chối sẽ không được đưa vào mô hình bệnh.",
            ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(flow, dpi=220, bbox_inches="tight")
    plt.close(fig)

    evaluation = ASSETS / "evaluation_protocol.png"
    fig, ax = plt.subplots(figsize=(10.5, 2.8))
    ax.axis("off")
    labels = ["Dữ liệu PlantDoc", "Chuẩn hóa\n& chia split", "Train + validation", "Khóa checkpoint\n& cấu hình", "Test cuối\n+ báo cáo"]
    for index, text in enumerate(labels):
        x = 0.07 + index * 0.215
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#EAF2F8", "edgecolor": "#2E75B6"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + 0.12, 0.55), xytext=(x + 0.08, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.text(0.5, 0.13, "Không chọn ngưỡng, checkpoint hoặc mô hình bằng tập test.", ha="center", fontsize=9, color="#9C0006")
    fig.tight_layout()
    fig.savefig(evaluation, dpi=220, bbox_inches="tight")
    plt.close(fig)

    tests = ASSETS / "test_status.png"
    names = ["Python backend", "Frontend", "Production build"]
    values = [35, 9, 1]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bars = ax.barh(names, values, color=["#70AD47", "#5B9BD5", "#ED7D31"])
    ax.set_xlim(0, 40)
    ax.set_xlabel("Số kiểm tra đạt / trạng thái build")
    ax.set_title("Kết quả kiểm thử phần mềm đã chạy")
    for bar, label in zip(bars, ["35 passed", "9 passed", "Build pass"]):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(tests, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return flow, evaluation, tests


def clear_paragraph(paragraph) -> None:
    element = paragraph._element
    for child in list(element):
        if child.tag != qn("w:pPr"):
            element.remove(child)


def set_text(paragraph, text: str) -> None:
    clear_paragraph(paragraph)
    paragraph.add_run(text)


def insert_after(paragraph, new_element) -> None:
    paragraph._p.addnext(new_element)


def insert_paragraph_after(paragraph, text: str, style: str | None = None):
    from docx.text.paragraph import Paragraph

    new = OxmlElement("w:p")
    insert_after(paragraph, new)
    result = Paragraph(new, paragraph._parent)
    result.style = style or paragraph.style
    result.add_run(text)
    return result


def insert_figure_after(paragraph, image: Path, caption: str):
    from docx.text.paragraph import Paragraph

    picture_xml = OxmlElement("w:p")
    insert_after(paragraph, picture_xml)
    picture = Paragraph(picture_xml, paragraph._parent)
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(image), width=Cm(15.7))
    caption_p = insert_paragraph_after(picture, caption, "4_Figures_c6")
    caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return caption_p


def replace_table(table, headers: list[str], rows: list[list[str]]) -> None:
    while len(table.rows) > 1:
        table._tbl.remove(table.rows[-1]._tr)
    header = table.rows[0].cells
    for cell, value in zip(header, headers, strict=True):
        cell.text = value
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row, strict=True):
            cell.text = value


def trim_table_columns(table, count: int) -> None:
    """Remove trailing Word grid columns so the result table stays readable on A4."""
    for row in table.rows:
        while len(row.cells) > count:
            row._tr.remove(row.cells[-1]._tc)
    grid = table._tbl.tblGrid
    while len(grid.gridCol_lst) > count:
        grid.remove(grid.gridCol_lst[-1])


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, OUTPUT)
    flow, evaluation, tests = make_figures()
    document = Document(OUTPUT)
    paragraphs = document.paragraphs

    # Correct the architecture claims in Chapter 2 to this project's implementation.
    set_text(
        paragraphs[294],
        "2.3. Kiến trúc mô hình và thiết lập triển khai YOLO11s",
    )
    set_text(
        paragraphs[295],
        "2.3.1. Mô hình chính YOLO11s và nguyên tắc fine-tune từ COCO",
    )
    set_text(
        paragraphs[311],
        "2.3.2. Cấu hình huấn luyện và nguyên tắc tái lập thí nghiệm",
    )
    set_text(
        paragraphs[312],
        "Mô hình chính của hệ thống là YOLO11s khởi tạo từ trọng số COCO. Cấu hình được lưu trong tệp YAML, bao gồm seed, kích thước ảnh, batch, số epoch, patience, device và đường dẫn dữ liệu. Checkpoint chỉ được ghi sau khi quá trình train hoàn tất; không tự tải trọng số khi server khởi động.",
    )
    set_text(
        paragraphs[318],
        "2.4. Thiết kế luồng xử lý và giao diện web local",
    )
    set_text(
        paragraphs[320],
        "Hệ thống web local gồm React + Vite + TypeScript ở frontend và FastAPI ở backend. Ảnh upload dùng REST; webcam dùng WebSocket với JPEG, frame_id và cơ chế latest-frame-wins. Tất cả suy luận mặc định chạy tại 127.0.0.1, không gửi ảnh ra Internet.",
    )
    set_text(
        paragraphs[321],
        "Bước 1 - Kiểm tra ảnh đầu vào: Backend giải mã ảnh, giới hạn kích thước và kiểm tra định dạng. Một tomato gate độc lập có thể xác nhận lá cà chua trước detector. Khi gate từ chối, hệ thống trả thông báo nhập ảnh khác và không chạy nhận diện bệnh; khi chưa có checkpoint gate, trạng thái phải là unverified thay vì kết luận sai.",
    )
    set_text(
        paragraphs[322],
        "Bước 2 - Suy luận: YOLO11s nhận ảnh RGB và trả bounding box chuẩn hóa 0..1, nhãn thuộc đúng 9 lớp cà chua và score. ModelManager chỉ cho phép model trong allowlist và báo MODEL_UNAVAILABLE khi thiếu checkpoint local.",
    )
    set_text(
        paragraphs[323],
        "Bước 3 - Hậu xử lý: Confidence, IoU và max_det được kiểm tra ở API. Kết quả có image width/height, inference_ms và danh sách box chuẩn hóa. Frontend vẽ box theo object-fit để không lệch khi letterbox.",
    )
    set_text(
        paragraphs[324],
        "Bước 4 - Hiển thị và hướng xử lý: Giao diện hiển thị từng nhãn phát hiện và gợi ý quản lý riêng theo lớp bệnh. Khuyến nghị có mục đích học thuật, không thay thế xác nhận thực địa hoặc hướng dẫn sử dụng thuốc.",
    )
    set_text(
        paragraphs[325],
        "2.4.2. Kiểm tra loại cây và hướng xử lý theo lớp bệnh",
    )
    set_text(
        paragraphs[326],
        "YOLO bệnh lá được huấn luyện trên các lớp cà chua nên không thể tự suy ra đáng tin cậy rằng một lá tương tự thuộc cây khác. Vì vậy hệ thống tách tomato gate thành mô hình nhị phân tomato_leaf/other. Checkpoint gate cần có ảnh lá cà chua dương tính và ảnh âm tính đa dạng như lá cây khác, đất, cỏ, tay và nền. Không dùng heuristic màu xanh để suy đoán loại cây.",
    )
    set_text(
        paragraphs[327],
        "Nếu phát hiện early blight hoặc septoria, ứng dụng đề xuất vệ sinh lá bệnh, giảm ẩm trên lá và theo dõi cây lân cận. Nếu phát hiện late blight hoặc virus, giao diện nhắc kiểm tra toàn ruộng và xác nhận sớm với cán bộ chuyên môn vì nguy cơ lây lan cao.",
    )
    set_text(
        paragraphs[328],
        "Với bacterial spot và leaf mold, khuyến nghị tập trung vào dụng cụ sạch, thông gió và tránh tưới phun. Với spider mites, ứng dụng gợi ý kiểm tra mặt dưới lá và bảo tồn thiên địch. Các gợi ý này không chứa liều lượng hoặc chỉ định thuốc.",
    )
    set_text(
        paragraphs[329],
        "Mỗi gợi ý phải được hiểu là hỗ trợ quan sát. Người dùng cần kiểm tra mẫu lá, điều kiện canh tác và quy định tại địa phương trước khi áp dụng biện pháp quản lý.",
    )
    chapter2_caption = insert_figure_after(
        paragraphs[330], flow, "Hình 2.7: Luồng xử lý local và cơ chế kiểm tra ảnh lá cà chua trước nhận diện bệnh"
    )

    # Replace unsupported quantitative narrative and create a truthful Chapter 3 results section.
    set_text(paragraphs[348], "3.2. Kết quả kiểm thử phần mềm và trạng thái thực nghiệm")
    set_text(paragraphs[349], "3.2.1. Kết quả kiểm thử backend và frontend")
    set_text(
        paragraphs[350],
        "Các kiểm thử tự động được chạy trong môi trường local sau khi hoàn thiện backend, WebSocket, evaluator và giao diện. Kết quả xác minh hợp đồng API, kiểm tra ảnh lỗi/quá lớn, model unavailable, giao thức frame_id, tọa độ box, response typing, cơ chế khuyến nghị theo lớp và build production.",
    )
    set_text(
        paragraphs[351],
        "Backend Python: pytest ghi nhận 35 kiểm thử đạt. Các test bao gồm pipeline dữ liệu, detector contract, REST API, WebSocket, evaluator, tomato gate và gợi ý theo lớp bệnh.",
    )
    set_text(
        paragraphs[352],
        "Frontend: Vitest ghi nhận 9 kiểm thử đạt; ESLint không phát hiện lỗi; Vite production build thành công. Node.js hiện có phiên bản 20.18.1 nên Vite đưa cảnh báo nâng lên 20.19+ hoặc 22.12+, dù build vẫn hoàn thành.",
    )
    set_text(
        paragraphs[353],
        "Kiểm thử graceful degradation: khi chưa có weights/yolo11s_coco/best.pt, /api/models trả available=false với lý do checkpoint local thiếu; API không tạo dự đoán giả. Đây là hành vi mong muốn trước khi có mô hình đã huấn luyện.",
    )
    test_caption = insert_figure_after(
        paragraphs[353], tests, "Hình 3.1: Tổng hợp kết quả kiểm thử phần mềm đã chạy")
    set_text(paragraphs[354], "3.2.2. Kết quả huấn luyện và đánh giá định lượng")
    set_text(
        paragraphs[355],
        "Tại thời điểm cập nhật báo cáo, chưa có checkpoint YOLO11s đã huấn luyện, prediction artifact hoặc metrics validation/test được lưu trong workspace. Vì vậy không báo cáo mAP@0.5, mAP@0.5:0.95, precision, recall, AP theo lớp, confusion matrix hay latency dưới dạng số để tránh số liệu không thể kiểm chứng.",
    )
    set_text(
        paragraphs[356],
        "Evaluator đã được cài đặt để tính mAP@0.5, mAP@0.5:0.95, AP theo lớp, precision, recall và confusion matrix từ COCO annotation cùng prediction thật. Test set chỉ mở sau khi checkpoint, dataset.yaml và kết quả validation đã được khóa bằng manifest.",
    )
    set_text(
        paragraphs[357],
        "Khi có dữ liệu và checkpoint, quy trình hợp lệ là: kiểm tra và chuẩn hóa PlantDoc; train YOLO11s; chạy validation để chọn checkpoint/ngưỡng; khóa cấu hình; chạy test một lần; sau đó sinh bảng và biểu đồ tự động. Mọi metric phải truy xuất được về artifact đã lưu.",
    )
    evaluation_caption = insert_figure_after(
        paragraphs[357], evaluation, "Hình 3.2: Quy trình đánh giá tránh rò rỉ tập kiểm thử")
    set_text(paragraphs[358], "3.2.3. Phân tích lỗi cần kiểm chứng sau khi có prediction thật")
    set_text(
        paragraphs[359],
        "Các nguy cơ lỗi cần được kiểm chứng bằng confusion matrix và gallery prediction gồm: nhầm early blight, septoria và bacterial spot ở giai đoạn sớm; bỏ sót lá bị che khuất; và ảnh hưởng của ánh sáng, nền phức tạp hoặc vùng bệnh nhỏ. Đây là giả thuyết phân tích, chưa phải kết quả định lượng của mô hình.",
    )
    set_text(
        paragraphs[360],
        "Domain gap cần được báo cáo riêng: ảnh PlantDoc thu thập từ Internet có thể khác webcam thực tế về camera, ánh sáng, giống cây, nền, khoảng cách và giai đoạn bệnh. Kết quả tốt trên split PlantDoc không tự chứng minh hiệu quả ngoài ruộng.",
    )
    set_text(
        paragraphs[361],
        "Tomato gate giảm rủi ro detector gán bệnh cà chua cho lá cây khác, nhưng chỉ có giá trị khi checkpoint nhị phân được huấn luyện và đánh giá trên ảnh âm tính thật. Trước thời điểm đó, ứng dụng phải hiển thị unverified.",
    )
    set_text(paragraphs[362], "3.3. Tích hợp ứng dụng web local")
    set_text(paragraphs[363], "3.3.1. Kết quả tích hợp upload, webcam và API")
    set_text(
        paragraphs[364],
        "Ứng dụng web local đã có trang upload ảnh, camera trực tiếp, chọn model, điều chỉnh confidence/IoU, overlay bounding box, danh sách kết quả, FPS và inference latency. FastAPI cung cấp health, classes, models, detect và WebSocket detect; tất cả mặc định bind 127.0.0.1.",
    )
    set_text(
        paragraphs[365],
        "Với webcam, browser gửi JPEG kèm frame_id. Server trả kết quả cùng frame_id; client bỏ kết quả cũ để tránh vẽ box lên nhầm khung hình. Hệ thống giới hạn payload, ảnh lỗi và cấu hình sai để không làm server dừng.",
    )
    set_text(paragraphs[366], "3.3.2. Giới hạn của bản demo")
    set_text(
        paragraphs[367],
        "Live preview hoàn chỉnh chỉ được xác nhận sau khi có checkpoint YOLO11s hợp lệ và kiểm thử bằng webcam thật. Hiện backend khởi động được khi thiếu checkpoint nhưng trả MODEL_UNAVAILABLE thay vì suy luận giả.",
    )
    set_text(
        paragraphs[368],
        "Không diễn giải bounding box là diện tích đốm bệnh hoặc mức độ bệnh. Annotation PlantDoc mô tả lá/cụm lá; hệ thống chỉ hỗ trợ khoanh vùng và nhãn tham khảo.",
    )
    set_text(paragraphs[369], "3.3.3. Hướng hoàn thiện thực nghiệm")
    set_text(
        paragraphs[370],
        "Cần bổ sung checkpoint YOLO11s đã fine-tune, checkpoint tomato gate, tập ảnh cà chua thực địa độc lập có nhãn và đo latency trên cùng một thiết bị. Sau đó cập nhật tự động bảng metric, confusion matrix, ảnh demo và báo cáo phân tích lỗi.",
    )
    set_text(paragraphs[371], "3.4. Bảng trạng thái kết quả và khả năng tái lập")
    set_text(
        paragraphs[372],
        "Bảng dưới đây thay thế các số liệu hiệu năng chưa có artifact. Giá trị định lượng chỉ được điền khi lệnh đánh giá đã tạo metrics JSON tương ứng.",
    )
    set_text(paragraphs[373], "")
    set_text(paragraphs[374], "")
    set_text(paragraphs[375], "")
    set_text(paragraphs[376], "")
    set_text(paragraphs[377], "")
    set_text(paragraphs[378], "")
    set_text(paragraphs[379], "")
    set_text(paragraphs[380], "")
    set_text(paragraphs[381], "")
    set_text(paragraphs[382], "")
    set_text(paragraphs[384], "3.4.1. Bảng trạng thái huấn luyện, đánh giá và kiểm thử")
    set_text(paragraphs[385], "Bảng 3.1: Trạng thái kết quả có thể kiểm chứng tại thời điểm cập nhật")
    set_text(paragraphs[386], "")
    set_text(paragraphs[387], "")
    set_text(paragraphs[388], "")

    trim_table_columns(document.tables[9], 4)
    replace_table(
        document.tables[9],
        ["Hạng mục", "Trạng thái hiện tại", "Bằng chứng", "Bước tiếp theo"],
        [
            ["Huấn luyện YOLO11s", "Chưa chạy", "Chưa có best.pt", "tdet train --exp yolo11s_coco"],
            ["Validation", "Chưa có metric", "Chưa có prediction JSON", "tdet eval --split val"],
            ["Test cuối", "Chưa chạy", "Chỉ mở sau evaluation lock", "freeze-eval rồi eval-artifacts --split test"],
            ["Backend", "Đã kiểm thử", "pytest -q: 35 passed", "Chạy lại sau mỗi thay đổi"],
            ["Frontend", "Đã kiểm thử", "9 passed; lint và build pass", "Nâng Node lên 20.19+ hoặc 22.12+"],
            ["Tomato gate", "Chưa có checkpoint", "Trả unverified khi thiếu gate", "Train binary tomato_leaf/other"],
        ],
    )
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
